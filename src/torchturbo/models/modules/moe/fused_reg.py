# fused_reg.py — 自研注册算子（FusedDispatchGmm + MoeCombine, CANN 9.2 重建）生产接入层
#
# 设计来源：linghe_exp/torch_bind_c92/INTEGRATION.md（P4 设计定稿）+
#           torch_bind/python/fused_moe.py（toy 全链路数值已验证的 fwd/bwd 逻辑）。
# 生产入口将 dispatch+fc1+SwiGLU+fc2+combine 及各自反向封装为一个
# autograd.Function；前向保存 compact 行计划供反向复用，通信窗口仍按 chunk 调用。
#
# 开关：YAML moe.fused_reg: true（不用环境变量）。
import ctypes
import os
import sys
import threading
import time

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from torchturbo.models.modules.comm.comm import all_to_all as autograd_all_to_all

_ext = None
_ext_lock = threading.Lock()
_op_libs = []

# EP64 与已验证的两级 dispatch 测试保持同一窗口口径。
_CHUNK_TOKENS = 512


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        ext_dir = os.environ.get(
            "TT_FUSED_REG_EXT", "/storage/lf02381138/linghe_exp/torch_bind_2l_c92/python"
        )
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        exp_root = os.path.dirname(os.path.dirname(ext_dir))
        opp_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        for package in ("p3_fused_op_2l_c92", "combine_2l_c92"):
            vendor = os.path.join(exp_root, package, "opp_install/vendors/customize")
            if vendor not in opp_paths:
                opp_paths.append(vendor)
            lib = os.path.join(vendor, "op_api/lib/libcust_opapi.so")
            _op_libs.append(ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL))
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(filter(None, opp_paths))
        import fused_moe_ext

        _ext = fused_moe_ext
    return _ext


def _get_hcomm(group):
    rank = dist.get_rank(group)
    global_rank = dist.get_global_rank(group, rank)
    return group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)


class _GroupCtx:
    __slots__ = ("world", "rank", "aiv", "aic", "tile_m", "mode", "group")

    def __init__(self, world, rank, aiv, aic, tile_m, mode, group):
        self.world, self.rank, self.aiv, self.aic, self.tile_m, self.mode, self.group = (
            world, rank, aiv, aic, tile_m, mode, group)


_CTXS = {}
_PLAN_CACHE_HITS = 0
_PLAN_CACHE_MISSES = 0


def _cached_plan_selected(experts, chunk_idx, expert_ids, valid_lengths):
    global _PLAN_CACHE_MISSES
    _PLAN_CACHE_MISSES += 1
    return None


def _cache_plan_selected(experts, chunk_idx, selected, expert_ids, valid_lengths):
    return None


def ensure_ctx(group, h, top_k, e_loc, aiv_num=8, aic_num=4, tile_m=128, buff_mb=512):
    """每 EP 组一次；HcclAllocComResourceByTiling 是集合式调用，前后必须组内对齐。"""
    key = (id(group), h, top_k, e_loc)
    if key in _CTXS:
        return _CTXS[key]
    ext = _load_ext()
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    ranks = dist.get_process_group_ranks(group)
    gid = min(ranks)
    gdir = os.path.join(os.environ.get("OUTPUT_DIR", "/tmp"), "mc2_groups", f"g{gid}")
    if rank == 0:
        import shutil
        shutil.rmtree(gdir, ignore_errors=True)
    dist.barrier(group)
    mode = 10 if world == 64 else 0
    if mode == 10:
        nnodes = world // 16
        node = rank // 16
        local = rank % 16
        node_dir = os.path.join(gdir, f"node{node}")
        lane_dir = os.path.join(gdir, f"lane{local}")
        os.makedirs(node_dir, exist_ok=True)
        os.makedirs(lane_dir, exist_ok=True)
        dist.barrier(group)
        name1 = ext.file_comm_init(node_dir, local, 16, buff_mb)
        name2 = ext.file_comm_init(lane_dir, node, nnodes, 1536)
        if rank == 0:
            print("[FULL_SINGLE] file comms initialized", flush=True)
        ext.init_dispatch_gmm_ctx(name1, _CHUNK_TOKENS, h, top_k, world, e_loc,
                                  aiv_num, buff_mb)
        if rank == 0:
            print("[FULL_SINGLE] comm1 context initialized", flush=True)
        dist.barrier(group)
        ext.init_dispatch_gmm_2l_ctx(name2, nnodes, 16)
        if rank == 0:
            print("[FULL_SINGLE] comm2 context initialized", flush=True)
    else:
        os.makedirs(gdir, exist_ok=True)
        dist.barrier(group)
        name = ext.file_comm_init(gdir, rank, world, buff_mb)
        ext.init_combine_ctx(name, _CHUNK_TOKENS, h, top_k, world, aiv_num)
        ext.init_dispatch_gmm_ctx(name, _CHUNK_TOKENS, h, top_k, world, e_loc,
                                  aiv_num, buff_mb)
    ext.init_dispatch_gmm_sync_ctx(_get_hcomm(group), rank)
    dist.barrier(group)
    ctx = _GroupCtx(world, rank, aiv_num, aic_num, tile_m, mode, group)
    _CTXS[key] = ctx
    return ctx


def _valid_selector(expand_idx, valid_lengths=None):
    src = expand_idx[:, 0].long()
    valid = src >= 0
    if valid_lengths is not None:
        token = expand_idx[:, 1].long()
        valid &= token < valid_lengths[src.clamp(min=0)]
    return valid.nonzero(as_tuple=False).squeeze(1)


def _select_valid(gmm_out, expand_idx, valid_lengths=None):
    sel = _valid_selector(expand_idx, valid_lengths)
    return gmm_out.index_select(0, sel), expand_idx.index_select(0, sel)


def _filter_dummy_counts(recv_counts, etn, valid_lengths, top_k):
    recv_counts = recv_counts.clone()
    etn = etn.clone()
    if dist.get_rank() == 0:
        dummy = (_CHUNK_TOKENS - valid_lengths) * top_k
        recv_counts.view(valid_lengths.numel(), -1)[:, 0] -= dummy.to(recv_counts.dtype)
        etn[0] -= dummy.sum().to(etn.dtype)
    return recv_counts, etn


def _recv_counts(send_l, device, world):
    counts = [torch.empty(world, dtype=torch.float32, device=device) for _ in range(world)]
    dist.all_to_all(counts, [torch.tensor(send_l, dtype=torch.float32, device=device)] * world)
    torch.npu.synchronize()
    rank = dist.get_rank()
    return [int(counts[s][rank].item()) for s in range(world)]


def _argsort_int(indices):
    return torch.argsort(indices.float())


def _segment_sum_rows(values, indices, bs):
    if indices.numel() == 0:
        return torch.zeros(bs, *values.shape[1:], device=values.device, dtype=values.dtype)
    order = _argsort_int(indices)
    sorted_idx = indices[order]
    zero = torch.zeros(1, *values.shape[1:], device=values.device, dtype=values.dtype)
    cumulative = torch.cat([zero, torch.cumsum(values[order], dim=0)], dim=0)
    positions = torch.arange(bs, device=values.device)
    ends = torch.searchsorted(sorted_idx, positions, right=True)
    starts = torch.searchsorted(sorted_idx, positions, right=False)
    return cumulative[ends] - cumulative[starts]


def _fetch_expand_x(x, expert_ids, idx_c, e_loc, actual_bs):
    bs, top_k = expert_ids.shape
    h = x.shape[1]
    world = dist.get_world_size()
    device = x.device
    dst = (expert_ids // e_loc).reshape(-1)
    tt = torch.arange(bs, device=device).repeat_interleave(top_k)
    kk = torch.arange(top_k, device=device).repeat(bs)
    valid = tt < actual_bs
    valid_dst = dst[valid]
    send_l = torch.bincount(valid_dst, minlength=world).cpu().tolist()
    recv_l = _recv_counts(send_l, device, world)
    order = _argsort_int(valid_dst)
    send_keys = torch.stack([tt[valid], kk[valid]], dim=1).to(torch.int32)[order].contiguous()
    send_x = x[tt[valid]][order].contiguous()
    recv_keys = torch.empty(sum(recv_l), 2, device=device, dtype=torch.int32)
    recv_x = torch.empty(sum(recv_l), h, device=device, dtype=x.dtype)
    dist.all_to_all_single(recv_keys, send_keys, output_split_sizes=recv_l,
                           input_split_sizes=send_l)
    dist.all_to_all_single(recv_x, send_x, output_split_sizes=recv_l,
                           input_split_sizes=send_l)
    recv_src = torch.repeat_interleave(
        torch.arange(world, device=device), torch.tensor(recv_l, device=device))
    recv_route_keys = (recv_src * bs + recv_keys[:, 0].long()) * top_k + recv_keys[:, 1].long()
    src_rank = idx_c[:, 0].long()
    t_idx = idx_c[:, 1].long()
    k_idx = idx_c[:, 2].long()
    idx_route_keys = (src_rank * bs + t_idx) * top_k + k_idx
    recv_order = _argsort_int(recv_route_keys)
    idx_order = _argsort_int(idx_route_keys)
    expand_x = torch.empty_like(recv_x)
    expand_x[idx_order] = recv_x[recv_order]
    return expand_x


def _reset_dispatch_2l(x, expert_ids, weight, group):
    _load_ext().dispatch_gmm(x, expert_ids, weight, 8, 4, 128, 12, 1)
    torch.npu.synchronize()
    dist.barrier(group)


def _fetch_expand_x_2l(x, expert_ids, idx_c, ep_recv_counts, e_loc, valid_lengths, group):
    ext = _load_ext()
    dummy_weight = torch.zeros(e_loc, x.shape[1], 1, device=x.device, dtype=x.dtype)
    _reset_dispatch_2l(x, expert_ids, dummy_weight, group)
    dummy_rows = int(((_CHUNK_TOKENS * dist.get_world_size(group) - valid_lengths.sum())
                      * expert_ids.shape[1]).item())
    recv_cap = max(idx_c.shape[0] + dummy_rows, 32768)
    expanded, idx, _etn, _recv_counts, _ep_counts, _ = ext.dispatch_gmm(
        x, expert_ids, dummy_weight, 8, 4, 128, 11, recv_cap)
    sel = _valid_selector(idx, valid_lengths)
    torch.npu.synchronize()
    dist.barrier(group)
    return expanded.index_select(0, sel)


class _DispatchGmmFn(torch.autograd.Function):
    """fwd: FusedDispatchGmm（dispatch + fc1 GMM）。bwd: dgrad/wgrad + a2a 回源聚合。

    反向逻辑 = torch_bind toy（数值已验证）：
      grad_W[e] = expandX_c[e]^T @ grad_gmmC[e]（本地）
      grad_expandX_c[e] = grad_gmm_c[e] @ W_e^T
      grad_x：行打包 (t, grad) a2a 回源 rank，index_add 聚合
    """

    @staticmethod
    def forward(ctx, x, expert_ids, weight, group_key, valid_lengths=None, actual_bs=None):
        ext = _load_ext()
        ctx.actual_bs = x.shape[0] if actual_bs is None else actual_bs
        ctxG = _CTXS[group_key]
        plain_weight = weight._local_tensor if hasattr(weight, "_local_tensor") else weight
        with torch.no_grad():
            if ctxG.mode == 10:
                _reset_dispatch_2l(x, expert_ids, plain_weight, ctxG.group)
            (expand_x, expand_idx, etn, _recv_counts, ep_recv_counts,
             gmm_out) = ext.dispatch_gmm(x, expert_ids, plain_weight, ctxG.aiv, ctxG.aic,
                                         ctxG.tile_m, ctxG.mode)
            if ctxG.mode == 10:
                _recv_counts, etn = _filter_dummy_counts(
                    _recv_counts, etn, valid_lengths, expert_ids.shape[1])
            sel = _valid_selector(expand_idx, valid_lengths)
            gmm_c = gmm_out.index_select(0, sel)
            idx_c = expand_idx.index_select(0, sel)
        combine_counts = (_recv_counts.view(ctxG.world, -1).sum(dim=1).to(torch.int32)
                          if ctxG.mode == 10 else ep_recv_counts)
        ctx.save_for_backward(x, expert_ids, weight, gmm_c, idx_c, etn, _recv_counts,
                              combine_counts, valid_lengths)
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.group_key = group_key
        ctx.weight_mesh = getattr(weight, "device_mesh", None)
        ctx.weight_placements = getattr(weight, "placements", None)
        return gmm_c, idx_c, etn, combine_counts

    @staticmethod
    def backward(ctx, grad_gmm_out, grad_expand_idx, grad_etn, grad_eprc):
        (x, expert_ids, weight, gmm_c, idx_c, etn, recv_counts,
         ep_recv_counts, valid_lengths) = ctx.saved_tensors
        bs, h = ctx.x_shape
        plain_weight = weight._local_tensor if hasattr(weight, "_local_tensor") else weight
        e_loc = plain_weight.shape[0]
        device = plain_weight.device
        world = dist.get_world_size()

        grad_gmm_c = grad_gmm_out.float().contiguous()
        t_idx = idx_c[:, 1].long()
        src_rank = idx_c[:, 0].long()
        if _CTXS[ctx.group_key].mode == 10:
            expand_x_c = _fetch_expand_x_2l(x, expert_ids, idx_c, ep_recv_counts, e_loc,
                                            valid_lengths, _CTXS[ctx.group_key].group)
        else:
            expand_x_c = _fetch_expand_x(x, expert_ids, idx_c, e_loc, ctx.actual_bs)

        grad_w = torch.zeros(e_loc, plain_weight.shape[1], plain_weight.shape[2], device=device,
                             dtype=torch.float32)
        grad_expand_x_c = torch.zeros(gmm_c.shape[0], expand_x_c.shape[1],
                                      device=device, dtype=torch.float32)
        expert_of_row = torch.repeat_interleave(torch.arange(e_loc, device=device),
                                                etn.long())
        for e in range(e_loc):
            m = expert_of_row == e
            if not m.any():
                continue
            w_e = plain_weight[e].float()
            grad_expand_x_c[m] = grad_gmm_c[m] @ w_e.t()
            grad_w[e] = expand_x_c[m].float().t() @ grad_gmm_c[m]

        # a2a 回源：行打包 (t, grad) 按 srcRank 分组
        send_l = recv_counts.view(world, -1).sum(dim=1).cpu().tolist()
        recv_l = _recv_counts(send_l, device, world)
        order = _argsort_int(src_rank)
        pack = torch.cat([t_idx.unsqueeze(1).float(), grad_expand_x_c],
                         dim=1)[order].contiguous()
        recv_buf = torch.empty(sum(recv_l), 1 + h, device=device, dtype=torch.float32)
        dist.all_to_all_single(recv_buf, pack, output_split_sizes=recv_l,
                               input_split_sizes=send_l)
        grad_x = _segment_sum_rows(recv_buf[:, 1:], recv_buf[:, 0].long(), bs)
        grad_w = grad_w.to(weight.dtype)
        if ctx.weight_mesh is not None:
            grad_w = DTensor.from_local(grad_w, ctx.weight_mesh, ctx.weight_placements)
        return grad_x.to(ctx.x_dtype), None, grad_w, None, None, None


def _fused_combine_2l(fc2_out, idx_c, ep_recv_counts, expert_scales, valid_lengths, e_loc, group):
    ext = _load_ext()
    nnodes = dist.get_world_size(group) // 16
    for phase in (100, 0, 1, 2):
        x_out = ext.moe_combine_2l(fc2_out, idx_c, ep_recv_counts, expert_scales,
                                   valid_lengths, 8, nnodes, 16, phase)
        torch.npu.synchronize()
        dist.barrier(group)
    return x_out


def _expert_padded_indices(etn, e_loc, total_rows, device):
    counts = etn.long()
    experts = torch.repeat_interleave(torch.arange(e_loc, device=device), counts)
    compact_start = counts.cumsum(0) - counts
    padded_counts = ((counts + 127) // 128) * 128
    padded_start = padded_counts.cumsum(0) - padded_counts
    local_rows = torch.arange(total_rows, device=device) - compact_start[experts]
    return padded_start[experts] + local_rows


def _fused_fc2_combine_2l(act, idx_c, ep_recv_counts, expert_scales, valid_lengths,
                          etn, w2, e_loc, group, return_fc2=False):
    ext = _load_ext()
    nnodes = dist.get_world_size(group) // 16
    padded_indices = _expert_padded_indices(etn, e_loc, act.shape[0], act.device)
    act_padded = act.new_zeros((act.shape[0] + e_loc * 128, act.shape[1]))
    act_padded.index_copy_(0, padded_indices, act)
    fc2_padded = ext.moe_fc2_gmm(act_padded, w2, etn)
    fc2_out = fc2_padded.index_select(0, padded_indices)
    torch.npu.synchronize()
    dist.barrier(group)
    for phase in (100, 0, 1, 2):
        x_out = ext.moe_combine_2l(fc2_out, idx_c, ep_recv_counts, expert_scales,
                                   valid_lengths, 8, nnodes, 16, phase)
        torch.npu.synchronize()
        dist.barrier(group)
    if return_fc2:
        return x_out, fc2_out
    return x_out


def _fused_combine_2l_backward(grad_out, fc2_out, idx_c, ep_recv_counts, expert_scales,
                               valid_lengths, expert_ids, e_loc, group):
    ext = _load_ext()
    nnodes = dist.get_world_size(group) // 16
    world = dist.get_world_size(group)
    h = fc2_out.shape[1]
    top_k = expert_scales.shape[1]
    grad_out = grad_out.contiguous().to(fc2_out.dtype)
    if grad_out.shape[0] < expert_scales.shape[0]:
        pad = expert_scales.shape[0] - grad_out.shape[0]
        grad_out = torch.cat([grad_out, grad_out.new_zeros(pad, h)], dim=0)
    dummy_weight = torch.zeros(e_loc, h, 1, device=fc2_out.device, dtype=fc2_out.dtype)
    dummy_rows = int(((_CHUNK_TOKENS * world - valid_lengths.sum()) * top_k).item())
    recv_cap = max(idx_c.shape[0] + dummy_rows, 32768)

    def dispatch_values(values):
        _reset_dispatch_2l(values, expert_ids, dummy_weight, group)
        expanded, idx, etn, recv_counts, _ep_counts, _ = ext.dispatch_gmm(
            values, expert_ids, dummy_weight, 8, 4, 128, 11, recv_cap)
        recv_counts, etn = _filter_dummy_counts(recv_counts, etn, valid_lengths, top_k)
        sel = _valid_selector(idx, valid_lengths)
        counts = recv_counts.view(world, -1).sum(dim=1).to(torch.int32)
        torch.npu.synchronize()
        dist.barrier(group)
        return expanded.index_select(0, sel), idx.index_select(0, sel), counts

    grad_scales = torch.zeros(fc2_out.shape[0] + expert_scales.numel(),
                              device=fc2_out.device, dtype=torch.float32)
    scale_routes = torch.zeros_like(grad_out)
    scale_routes[:, :top_k] = expert_scales.to(fc2_out.dtype)
    scale_expand, scale_idx, _ = dispatch_values(scale_routes)
    route_k = scale_idx[:, 2].long().unsqueeze(1)
    selected_scales = scale_expand.gather(1, route_k).float().squeeze(1)
    grad_scales[:fc2_out.shape[0]] = selected_scales
    del scale_expand, scale_idx, scale_routes, route_k

    grad_expand, idx_c, combine_counts = dispatch_values(grad_out)
    grad_input = torch.zeros_like(fc2_out)
    valid_lengths_i32 = valid_lengths.to(torch.int32)
    for phase in range(2, 6):
        if phase == 3:
            ext.moe_combine_2l(grad_expand, idx_c, combine_counts, expert_scales.float(),
                               valid_lengths, 8, nnodes, 16, 100)
            torch.npu.synchronize()
            dist.barrier(group)
        ext.moe_combine_2l_grad(grad_expand, idx_c, combine_counts, expert_scales.float(),
                                valid_lengths_i32, expert_ids, fc2_out, grad_input, grad_scales,
                                e_loc, 8, nnodes, 16, phase, 1)
        torch.npu.synchronize()
        dist.barrier(group)
    return grad_input, grad_scales[fc2_out.shape[0]:].view_as(expert_scales)


def _torch_combine(fc2_out, idx_c, expert_scales, recv_counts=None):
    device = fc2_out.device
    bs = expert_scales.shape[0]
    h = fc2_out.shape[1]
    world = dist.get_world_size()
    t_idx = idx_c[:, 1].long()
    src_rank = idx_c[:, 0].long()
    send_l = (recv_counts.view(world, -1).sum(dim=1) if recv_counts is not None else
              torch.bincount(src_rank, minlength=world)).cpu().tolist()
    recv_l = _recv_counts(send_l, device, world)
    k_idx = idx_c[:, 2].long()
    all_scales = [torch.empty_like(expert_scales) for _ in range(world)]
    dist.all_gather(all_scales, expert_scales)
    torch.npu.synchronize()
    all_scales = torch.stack(all_scales)
    scales = all_scales[src_rank, t_idx, k_idx].float().unsqueeze(1)
    if sum(send_l) != idx_c.shape[0]:
        dump_path = os.path.join(os.environ.get("OUTPUT_DIR", "/tmp"),
                                 f"count_mismatch_rank{dist.get_rank()}.pt")
        torch.save({"recv_counts": recv_counts.cpu(), "expand_idx": idx_c.cpu()}, dump_path)
        raise RuntimeError(f"dispatch count mismatch: {sum(send_l)} != {idx_c.shape[0]}")
    weighted = fc2_out.float() * scales
    pack = torch.cat([t_idx.float().unsqueeze(1), weighted], dim=1)
    pack = pack[_argsort_int(src_rank)].contiguous()
    recv_buf = torch.empty(sum(recv_l), 1 + h, device=device, dtype=torch.float32)
    dist.all_to_all_single(recv_buf, pack, output_split_sizes=recv_l,
                           input_split_sizes=send_l)
    torch.npu.synchronize()
    x_out = _segment_sum_rows(recv_buf[:, 1:], recv_buf[:, 0].long(), bs)
    return x_out.to(fc2_out.dtype)


class _Fc2CombineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, act, idx_c, ep_recv_counts, expert_scales, need_dst, group_key,
                actual_bs, valid_lengths, expert_ids, e_loc, etn, w2):
        ctxG = _CTXS[group_key]
        plain_weight = w2._local_tensor if isinstance(w2, DTensor) else w2
        with torch.no_grad():
            x_out, fc2_out = _fused_fc2_combine_2l(
                act, idx_c, ep_recv_counts, expert_scales, valid_lengths, etn, plain_weight,
                e_loc, ctxG.group, return_fc2=True)
        ctx.save_for_backward(act, idx_c, ep_recv_counts, expert_scales, expert_ids,
                              valid_lengths, etn, w2, fc2_out)
        ctx.group_key = group_key
        ctx.e_loc = e_loc
        ctx.weight_mesh = getattr(w2, "device_mesh", None)
        ctx.weight_placements = getattr(w2, "placements", None)
        return x_out

    @staticmethod
    def backward(ctx, grad_out):
        (act, idx_c, ep_recv_counts, expert_scales, expert_ids, valid_lengths,
         etn, w2, fc2_out) = ctx.saved_tensors
        ctxG = _CTXS[ctx.group_key]
        grad_fc2, grad_scales = _fused_combine_2l_backward(
            grad_out, fc2_out, idx_c, ep_recv_counts, expert_scales, valid_lengths,
            expert_ids, ctx.e_loc, ctxG.group)
        plain_weight = w2._local_tensor if hasattr(w2, "_local_tensor") else w2
        grad_act, grad_weight = _load_ext().moe_fc2_gmm_grad(
            act, plain_weight, grad_fc2.contiguous(), etn)
        if ctx.weight_mesh is not None:
            grad_weight = DTensor.from_local(grad_weight, ctx.weight_mesh, ctx.weight_placements)
        return (grad_act[:act.shape[0]], None, None, grad_scales.to(expert_scales.dtype),
                None, None, None, None, None, None, None, grad_weight)


class _CombineFn(torch.autograd.Function):
    """fwd: MoeCombine（a2a 回源 + 加权求和）。bwd: toy 反向（a2a 双程）。"""

    @staticmethod
    def forward(ctx, fc2_out, idx_c, ep_recv_counts, expert_scales, need_dst, group_key,
                actual_bs=None, valid_lengths=None, expert_ids=None, e_loc=None):
        ctxG = _CTXS[group_key]
        with torch.no_grad():
            if ctxG.mode == 10 and valid_lengths is not None:
                x_out = _fused_combine_2l(fc2_out, idx_c, ep_recv_counts, expert_scales,
                                          valid_lengths, e_loc, ctxG.group)
            elif ctxG.mode == 10:
                x_out = _torch_combine(fc2_out, idx_c, expert_scales, ep_recv_counts)
            else:
                ext = _load_ext()
                x_out = ext.moe_combine(fc2_out.to(torch.float16), idx_c, ep_recv_counts,
                                        expert_scales.float(), ctxG.aiv, 7)
        ctx.save_for_backward(fc2_out, idx_c, ep_recv_counts, expert_scales, need_dst,
                              expert_ids, valid_lengths)
        ctx.actual_bs = expert_scales.shape[0] if actual_bs is None else actual_bs
        ctx.group_key = group_key
        ctx.e_loc = e_loc
        return x_out

    @staticmethod
    def backward(ctx, grad_out):
        (fc2_out, idx_c, ep_recv_counts, expert_scales, need_dst,
         expert_ids, valid_lengths) = ctx.saved_tensors
        ctxG = _CTXS[ctx.group_key]
        if ctxG.mode == 10 and valid_lengths is not None:
            grad_input, grad_scales = _fused_combine_2l_backward(
                grad_out, fc2_out, idx_c, ep_recv_counts, expert_scales, valid_lengths,
                expert_ids, ctx.e_loc, ctxG.group)
            return (grad_input, None, None, grad_scales.to(expert_scales.dtype),
                    None, None, None, None, None, None)
        device = fc2_out.device
        h = fc2_out.shape[1]
        bs = expert_scales.shape[0]
        top_k = expert_scales.shape[1]
        world = dist.get_world_size()

        grad_out = grad_out.contiguous().float()
        t_idx = idx_c[:, 1].long()
        k_idx = idx_c[:, 2].long()
        src_rank = idx_c[:, 0].long()

        # 第 1 程：gradOut + scales 从属主 rank 到专家 rank（"combine 的反向是 dispatch 结构"）
        tt = torch.arange(bs, device=device).repeat_interleave(top_k)
        kk = torch.arange(top_k, device=device).repeat(bs)
        valid = tt < ctx.actual_bs
        valid_dst = need_dst[valid]
        send_l = torch.bincount(valid_dst, minlength=world).cpu().tolist()
        recv_l = _recv_counts(send_l, device, world)
        tt = tt[valid]
        kk = kk[valid]
        order = _argsort_int(valid_dst)
        pack = torch.cat([tt.float().unsqueeze(1), kk.float().unsqueeze(1),
                          expert_scales.float()[tt, kk].unsqueeze(1),
                          grad_out[tt]], dim=1)[order].contiguous()
        recv_buf = torch.empty(sum(recv_l), 3 + h, device=device, dtype=torch.float32)
        dist.all_to_all_single(recv_buf, pack, output_split_sizes=recv_l,
                               input_split_sizes=send_l)

        n_rows = idx_c.shape[0]
        grad_rows = torch.zeros(n_rows, h, device=device, dtype=torch.float32)
        scale_rows = torch.zeros(n_rows, device=device, dtype=torch.float32)
        recv_src = torch.repeat_interleave(
            torch.arange(world, device=device), torch.tensor(recv_l, device=device))
        recv_keys = (recv_src * bs + recv_buf[:, 0].long()) * top_k + recv_buf[:, 1].long()
        idx_keys = (src_rank * bs + t_idx) * top_k + k_idx
        recv_order = _argsort_int(recv_keys)
        idx_order = _argsort_int(idx_keys)
        grad_rows[idx_order] = recv_buf[recv_order, 3:]
        scale_rows[idx_order] = recv_buf[recv_order, 2]

        grad_input = grad_rows * scale_rows.unsqueeze(1)  # grad wrt fc2_out
        grad_scales_contrib = (grad_rows * fc2_out.float()).sum(dim=1)

        # 第 2 程：grad_scales 贡献回属主
        send2_l = torch.bincount(src_rank, minlength=world).cpu().tolist()
        recv2_l = _recv_counts(send2_l, device, world)
        order2 = _argsort_int(src_rank)
        pack_s = torch.stack([t_idx.float(), k_idx.float(),
                              grad_scales_contrib], dim=1)[order2].contiguous()
        recv_s = torch.empty(sum(recv2_l), 3, device=device, dtype=torch.float32)
        dist.all_to_all_single(recv_s, pack_s, output_split_sizes=recv2_l,
                               input_split_sizes=send2_l)
        flat_idx = (recv_s[:, 0].long() * top_k + recv_s[:, 1].long())
        grad_scales = _segment_sum_rows(recv_s[:, 2], flat_idx, bs * top_k).reshape(bs, top_k)

        return (grad_input.to(fc2_out.dtype), None, None,
                grad_scales.to(expert_scales.dtype), None, None, None, None, None, None)


class _FullFusedMoeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, expert_ids, expert_scales, valid_lengths, weight1, weight2, group_key,
                swiglu_limit, chunk_idx, plan_selected, plan_source):
        ext = _load_ext()
        plain_w1 = weight1._local_tensor if hasattr(weight1, "_local_tensor") else weight1
        plain_w2 = weight2._local_tensor if hasattr(weight2, "_local_tensor") else weight2
        ctxG = _CTXS[group_key]
        with torch.no_grad():
            if plan_selected is None:
                outputs = ext.full_fused_moe(x, expert_ids, expert_scales, valid_lengths,
                                             plain_w1, plain_w2, ctxG.aiv, ctxG.aic, ctxG.tile_m,
                                             swiglu_limit, True)
            else:
                outputs = ext.full_fused_moe_with_plan(
                    x, expert_ids, expert_scales, valid_lengths, plain_w1, plain_w2,
                    ctxG.aiv, ctxG.aic, ctxG.tile_m, swiglu_limit, True, plan_selected)
        if plan_selected is None:
            _cache_plan_selected(plan_source, chunk_idx, outputs[9], expert_ids, valid_lengths)
        fc1_compact = outputs[6]
        act_compact = outputs[7]
        fc2_compact = outputs[8]
        ctx.save_for_backward(x, expert_ids, expert_scales, valid_lengths, weight1, weight2,
                              outputs[2], outputs[3], fc1_compact, act_compact, fc2_compact,
                              outputs[9])
        ctx.group_key = group_key
        ctx.swiglu_limit = swiglu_limit
        ctx.weight1_mesh = getattr(weight1, "device_mesh", None)
        ctx.weight1_placements = getattr(weight1, "placements", None)
        ctx.weight2_mesh = getattr(weight2, "device_mesh", None)
        ctx.weight2_placements = getattr(weight2, "placements", None)
        ctx.x_dtype = x.dtype
        ctx.scales_dtype = expert_scales.dtype
        ctx.actual_bs = x.shape[0]
        return outputs[0]

    _backward_calls = 0
    _backward_seconds = 0.0

    @staticmethod
    def backward(ctx, grad_out):
        backward_start = time.perf_counter()
        (x, expert_ids, expert_scales, valid_lengths, weight1, weight2,
         expand_idx, etn, fc1, act, fc2, plan_selected) = ctx.saved_tensors
        plain_w1 = weight1._local_tensor if hasattr(weight1, "_local_tensor") else weight1
        plain_w2 = weight2._local_tensor if hasattr(weight2, "_local_tensor") else weight2
        ctxG = _CTXS[ctx.group_key]
        grad_x, grad_scales, grad_w1, grad_w2 = _load_ext().full_fused_moe_backward(
            grad_out.contiguous(), x, expert_ids, expert_scales, valid_lengths,
            plain_w1, plain_w2, x, expand_idx, etn, fc1, act, fc2,
            ctxG.aiv, ctxG.aic, ctxG.tile_m, True, ctx.swiglu_limit, plan_selected)
        if ctx.weight1_mesh is not None:
            grad_w1 = DTensor.from_local(grad_w1, ctx.weight1_mesh, ctx.weight1_placements)
        if ctx.weight2_mesh is not None:
            grad_w2 = DTensor.from_local(grad_w2, ctx.weight2_mesh, ctx.weight2_placements)
        _FullFusedMoeFn._backward_calls += 1
        _FullFusedMoeFn._backward_seconds += time.perf_counter() - backward_start
        if dist.get_rank() == 0 and _FullFusedMoeFn._backward_calls % 320 == 0:
            print(
                f"[FUSED_TIME] backward_calls={_FullFusedMoeFn._backward_calls} "
                f"total_seconds={_FullFusedMoeFn._backward_seconds:.3f} "
                f"plan_cache_hits={_PLAN_CACHE_HITS} plan_cache_misses={_PLAN_CACHE_MISSES}",
                flush=True,
            )
        return (grad_x[:ctx.actual_bs].to(ctx.x_dtype), None,
                grad_scales[:ctx.actual_bs].to(ctx.scales_dtype), None, grad_w1, grad_w2,
                None, None, None, None, None)


class _ChunkContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _FullFusedLayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, expert_indices, probs, weight1, weight2, group_key,
                swiglu_limit):
        ctxG = _CTXS[group_key]
        T, H = hidden_states.shape
        top_k = expert_indices.shape[1]
        num_chunks = (T + _CHUNK_TOKENS - 1) // _CHUNK_TOKENS
        if ctxG.mode == 10:
            chunks = torch.tensor(num_chunks, device=hidden_states.device, dtype=torch.int32)
            dist.all_reduce(chunks, op=dist.ReduceOp.MAX)
            num_chunks = int(chunks.item())

        ctx.chunk_contexts = []
        ctx.actual_lengths = []
        ctx.weight1_mesh = getattr(weight1, "device_mesh", None)
        ctx.weight1_placements = getattr(weight1, "placements", None)
        ctx.weight2_mesh = getattr(weight2, "device_mesh", None)
        ctx.weight2_placements = getattr(weight2, "placements", None)
        ys = []
        for chunk_idx in range(num_chunks):
            s0 = chunk_idx * _CHUNK_TOKENS
            actual_bs = max(0, min(_CHUNK_TOKENS, T - s0))
            x_c = hidden_states[s0:s0 + actual_bs]
            ids_c = expert_indices[s0:s0 + actual_bs]
            p_c = probs[s0:s0 + actual_bs]
            valid_lengths = None
            if ctxG.mode == 10:
                length = torch.tensor([actual_bs], device=hidden_states.device, dtype=torch.int64)
                lengths = [torch.empty_like(length) for _ in range(ctxG.world)]
                dist.all_gather(lengths, length)
                valid_lengths = torch.cat(lengths)
                pad = _CHUNK_TOKENS - actual_bs
                if pad:
                    x_c = torch.cat([x_c, x_c.new_zeros(pad, H)], dim=0)
                    ids_c = torch.cat([ids_c, ids_c.new_zeros(pad, top_k)], dim=0)
                    p_c = torch.cat([p_c, p_c.new_zeros(pad, top_k)], dim=0)
            chunk_ctx = _ChunkContext()
            y_c = _FullFusedMoeFn.forward(
                chunk_ctx, x_c, ids_c, p_c, valid_lengths, weight1, weight2, group_key,
                swiglu_limit, chunk_idx, _cached_plan_selected(experts, chunk_idx, ids_c, valid_lengths),
                experts)
            chunk_ctx.weight1_mesh = None
            chunk_ctx.weight2_mesh = None
            ctx.chunk_contexts.append(chunk_ctx)
            ctx.actual_lengths.append(actual_bs)
            ys.append(y_c[:actual_bs])
        return torch.cat(ys, dim=0)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x_parts = []
        grad_scale_parts = []
        grad_w1 = None
        grad_w2 = None
        offset = 0
        for chunk_ctx, actual_bs in zip(ctx.chunk_contexts, ctx.actual_lengths):
            grads = _FullFusedMoeFn.backward(
                chunk_ctx, grad_out[offset:offset + actual_bs])
            grad_x_parts.append(grads[0][:actual_bs])
            grad_scale_parts.append(grads[2][:actual_bs])
            if grad_w1 is None:
                grad_w1 = grads[4].clone()
                grad_w2 = grads[5].clone()
            else:
                grad_w1.add_(grads[4])
                grad_w2.add_(grads[5])
            offset += actual_bs
        if ctx.weight1_mesh is not None:
            grad_w1 = DTensor.from_local(grad_w1, ctx.weight1_mesh, ctx.weight1_placements)
        if ctx.weight2_mesh is not None:
            grad_w2 = DTensor.from_local(grad_w2, ctx.weight2_mesh, ctx.weight2_placements)
        return (torch.cat(grad_x_parts, dim=0), None, torch.cat(grad_scale_parts, dim=0),
                grad_w1, grad_w2, None, None)


def fused_reg_moe_layer(hidden_states, expert_indices, probs, experts, group, config):
    T, H = hidden_states.shape
    expert_indices = expert_indices.to(dtype=torch.int32)
    top_k = expert_indices.shape[1]
    e_loc = experts.num_local_experts
    w1 = experts.weight1.view(e_loc, H, -1).to(
        device=hidden_states.device, dtype=hidden_states.dtype)
    w2 = experts.weight2.view(e_loc, -1, H).to(
        device=hidden_states.device, dtype=hidden_states.dtype)
    gkey = (id(group), H, top_k, e_loc)
    ensure_ctx(group, H, top_k, e_loc)
    return _FullFusedLayerFn.apply(
        hidden_states, expert_indices, probs, w1, w2, gkey,
        float(getattr(experts, "swiglu_limit", 0.0)))


def fused_reg_moe(hidden_states, expert_indices, probs, experts, group, config):
    """Run the fully fused dispatch+fc1+SwiGLU+fc2+combine operator and its backward.

    hidden_states: (T, H) 本 rank token；expert_indices: (T, topk) 全局专家号；
    probs: (T, topk) 路由权重（属主侧）；experts: GroupedMLP（持 w1/w2/activation）。
    返回 (T, H)。
    """
    forward_start = time.perf_counter()
    T, H = hidden_states.shape
    expert_indices = expert_indices.to(dtype=torch.int32)
    top_k = expert_indices.shape[1]
    e_loc = experts.num_local_experts
    # autograd 保留原始 FSDP 权重，仅在 kernel 调用内部解包。
    weight_start = time.perf_counter()
    w1s, w2s = experts.weight1, experts.weight2
    w1 = w1s.view(e_loc, H, -1).to(device=hidden_states.device, dtype=hidden_states.dtype)
    w2 = w2s.view(e_loc, -1, H).to(device=hidden_states.device, dtype=hidden_states.dtype)
    if dist.get_rank() == 0:
        print(
            f"[FUSED_TIME] layer={getattr(experts, 'layer_number', -1)} "
            f"weight_prepare_seconds={time.perf_counter() - weight_start:.6f}",
            flush=True,
        )
    gkey = (id(group), H, top_k, e_loc)
    ensure_ctx(group, H, top_k, e_loc)

    num_chunks = (T + _CHUNK_TOKENS - 1) // _CHUNK_TOKENS
    if _CTXS[gkey].mode == 10:
        chunks = torch.tensor(num_chunks, device=hidden_states.device, dtype=torch.int32)
        dist.all_reduce(chunks, op=dist.ReduceOp.MAX)
        num_chunks = int(chunks.item())

    ys = []
    for chunk_idx in range(num_chunks):
        s0 = chunk_idx * _CHUNK_TOKENS
        actual_bs = max(0, min(_CHUNK_TOKENS, T - s0))
        x_c = hidden_states[s0:s0 + actual_bs]
        ids_c = expert_indices[s0:s0 + actual_bs]
        p_c = probs[s0:s0 + actual_bs]
        valid_lengths = None
        if _CTXS[gkey].mode == 10:
            length = torch.tensor([actual_bs], device=hidden_states.device, dtype=torch.int64)
            lengths = [torch.empty_like(length) for _ in range(_CTXS[gkey].world)]
            dist.all_gather(lengths, length)
            valid_lengths = torch.cat(lengths)
            pad = _CHUNK_TOKENS - actual_bs
            if pad:
                x_c = torch.cat([x_c, x_c.new_zeros(pad, H)], dim=0)
                ids_c = torch.cat([ids_c, ids_c.new_zeros(pad, top_k)], dim=0)
                p_c = torch.cat([p_c, p_c.new_zeros(pad, top_k)], dim=0)
        y_c = _FullFusedMoeFn.apply(
            x_c, ids_c, p_c, valid_lengths, w1, w2, gkey,
            float(getattr(experts, "swiglu_limit", 0.0)), chunk_idx,
            _cached_plan_selected(experts, chunk_idx, ids_c, valid_lengths), experts)
        torch.npu.synchronize()
        dist.barrier(group)
        ys.append(y_c[:actual_bs])
    if dist.get_rank() == 0:
        print(
            f"[FUSED_TIME] layer={getattr(experts, 'layer_number', -1)} "
            f"forward_seconds={time.perf_counter() - forward_start:.6f}",
            flush=True,
        )
    return torch.cat(ys, dim=0)


def reference_reg_moe(hidden_states, expert_indices, probs, experts, group, config):
    """Differentiable native MoE used during checkpoint recomputation."""
    from torchturbo.ops import ops

    T, H = hidden_states.shape
    expert_indices = expert_indices.to(dtype=torch.int32)
    top_k = expert_indices.shape[1]
    e_loc = experts.num_local_experts
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    w1s, w2s = experts.weight1, experts.weight2
    w1 = w1s.view(e_loc, H, -1).to(device=hidden_states.device, dtype=hidden_states.dtype)
    w2 = w2s.view(e_loc, -1, H).to(device=hidden_states.device, dtype=hidden_states.dtype)
    gkey = (id(group), H, top_k, e_loc)
    if gkey not in _CTXS:
        _CTXS[gkey] = _GroupCtx(world, rank, 8, 4, 128, 10, group)

    num_chunks = (T + _CHUNK_TOKENS - 1) // _CHUNK_TOKENS
    chunks = torch.tensor(num_chunks, device=hidden_states.device, dtype=torch.int32)
    dist.all_reduce(chunks, op=dist.ReduceOp.MAX)
    num_chunks = int(chunks.item())
    ys = []
    for chunk_idx in range(num_chunks):
        s0 = chunk_idx * _CHUNK_TOKENS
        actual_bs = max(0, min(_CHUNK_TOKENS, T - s0))
        x_c = hidden_states[s0:s0 + actual_bs]
        ids_c = expert_indices[s0:s0 + actual_bs]
        p_c = probs[s0:s0 + actual_bs]
        need_dst = (ids_c // e_loc).reshape(-1)
        pad = _CHUNK_TOKENS - actual_bs
        if pad:
            x_c = torch.cat([x_c, x_c.new_zeros(pad, H)], dim=0)
            ids_c = torch.cat([ids_c, ids_c.new_zeros(pad, top_k)], dim=0)
            p_c = torch.cat([p_c, p_c.new_zeros(pad, top_k)], dim=0)
            need_dst = torch.cat([need_dst, need_dst.new_zeros(pad * top_k)])
        dst = ids_c // e_loc
        send_l = torch.bincount(dst.reshape(-1).cpu(), minlength=world).tolist()
        recv_l = _recv_counts(send_l, x_c.device, world)
        tt = torch.arange(_CHUNK_TOKENS, device=x_c.device).repeat_interleave(top_k)
        kk = torch.arange(top_k, device=x_c.device).repeat(_CHUNK_TOKENS)
        pack = torch.cat([tt.float().unsqueeze(1), kk.float().unsqueeze(1),
                          (ids_c % e_loc).float().reshape(-1, 1),
                          x_c.float().repeat_interleave(top_k, dim=0).reshape(-1, H)], dim=1)
        pack = pack[_argsort_int(dst.reshape(-1))].contiguous()
        recv = autograd_all_to_all(group, pack, recv_l, send_l)
        recv_src = torch.repeat_interleave(
            torch.arange(world, device=x_c.device), torch.tensor(recv_l, device=x_c.device))
        recv_e = recv[:, 2].long()
        keys = ((recv_e * world + recv_src) * (_CHUNK_TOKENS * top_k) +
                recv[:, 0].long() * top_k + recv[:, 1].long())
        order = _argsort_int(keys)
        recv = recv[order]
        recv_src = recv_src[order]
        expand_x = recv[:, 3:].to(hidden_states.dtype)
        idx_c = torch.stack([recv_src, recv[:, 0].long(), recv[:, 1].long()],
                            dim=1).to(torch.int32)
        count_keys = (recv_src * e_loc + recv_e[order]).cpu()
        recv_counts = torch.bincount(count_keys, minlength=world * e_loc).view(
            world, e_loc).to(x_c.device, dtype=torch.int32)
        etn = recv_counts.sum(dim=0).long()
        length = torch.tensor([actual_bs], device=x_c.device, dtype=torch.int64)
        lengths = [torch.empty_like(length) for _ in range(world)]
        dist.all_gather(lengths, length)
        valid_lengths = torch.cat(lengths)
        recv_counts, etn = _filter_dummy_counts(recv_counts, etn, valid_lengths, top_k)
        sel = _valid_selector(idx_c, valid_lengths)
        expand_x = expand_x.index_select(0, sel)
        idx_c = idx_c.index_select(0, sel)
        fc1 = ops.grouped_matmul(expand_x, w1, etn.cpu(), trans_b=False)
        act = experts.activation_func(fc1, None)
        fc2_out = ops.grouped_matmul(act, w2, etn.cpu(), trans_b=False)
        y_c = _CombineFn.apply(fc2_out, idx_c, recv_counts, p_c, need_dst, gkey, actual_bs)
        ys.append(y_c[:actual_bs])
    return torch.cat(ys, dim=0)
