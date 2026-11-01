"""Next-token cross-entropy loss through the output layer, on soft-capped logits z = c tanh(x W^T / c)
(arXiv 2408.00118 §2; paper §4): the mean over non-padding labels t of logsumexp(z) - z[t], computed CHUNK rows at a
time so the full logit matrix is never materialized. Training uses QuACK's (Dao-AILab/quack) fused linear cross
entropy with the soft-cap in its GEMM epilogue, which also produces the gradients; evaluation uses an fp32 GEMM."""

import functools

import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32
from quack.activation import tanh
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.library import identity_epi
from quack.epilogue.math import pexp2
from quack.epilogue.ops import ColVecLoad, ColVecReduce, ColVecSelect
from quack.epilogue.scaled_exp import LOG2E, MaxLog2
from quack.linear_cross_entropy import _lce_dw_scale, _lce_dx_scale, _lce_vscale_epi, lce_glue
from quack.operand_transform.host import transform_a_operand
from torch import Tensor
from torch.autograd.function import FunctionCtx

from corpus.loader import IGNORE_INDEX

# Rows per loss GEMM in training and evaluation: a multiple of 128 (the gradient GEMMs' tile_M) that divides
# MICRO_BATCH_TOKENS (98,304 = 12 x 8,192); among the sizes equally fast on a d24 micro-batch, the one using the least
# memory (an evaluation chunk's fp32 logits take 1 GiB). One chunk of 98,304 rows (E above 2^31 elements) faults in
# QuACK 0.6.5.
CHUNK = 8192
# Column tile of the logit GEMM, the granularity of k and sum_exp: a multiple of 64 (lce_glue's strips) dividing 2^15,
# at most 208 for the pingpong schedule
TILE_N1 = 128
# The dx/dW strip GEMMs' tile_N over C = 768: QuACK's rule (quack/linear_cross_entropy.py
# scaled_exp_linear_cross_entropy_fwd tile_n2), 192 as it divides C
TILE_N2 = 192
# Rows per program of cap_label_fix: one warp, one row per lane
FIX_ROWS = 32


class CappedMaxLog2(MaxLog2):
    """MaxLog2 of the capped logits: k = rne(c tanh(max / c) log2 e), valid since c tanh(z / c) is increasing."""

    def __init__(self, name: str, cap: float):
        super().__init__(name)
        self.cap = cap

    def config_key(self):
        return (*super().config_key(), self.cap)

    @cute.jit
    def stat_value(self, total, group_cols):
        return cute.math.roundeven(tanh(total * Float32(1 / self.cap)) * Float32(self.cap * LOG2E))


@functools.cache
def capped_scaled_exp_epi(cap: float):
    """scaled_exp_target_epi on z = c tanh(acc / c): D = 2^(z log2 e - k) (1 - tanh^2), sum_exp of 2^(z log2 e - k),
    the label's capped logit."""
    k_op, target = CappedMaxLog2("max_log2", cap), ColVecLoad("target")

    def prepass(acc):
        return {"max_log2": acc}

    @gemm_epilogue(ops={"max_log2": k_op}, prepass=prepass, prepass_outs=("max_log2",),
                   reduces={"sum_exp": ColVecReduce("sum_exp")},
                   outs={"target_logit": ColVecSelect("target_logit", idx_op=target)},
                   extra_ops=(k_op.out("max_log2_out"), target))
    def epi(acc, max_log2):
        t = tanh(acc * (1 / cap))
        z = t * cap
        e = pexp2(z * LOG2E - max_log2)
        return {"D": e * (1.0 - t * t), "sum_exp": e, "target_logit": z}

    return epi


@triton.jit
def cap_label_fix(E, target, zy, V, ignore_index, inv_cap, ROWS: tl.constexpr):
    """E[i, t_i] *= 1 - (zy_i / c)^2 for the valid rows: the cap's derivative at the label entry lce_glue wrote."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    t = tl.load(target + rows)
    valid = t != ignore_index
    tanh_y = tl.load(zy + rows) * inv_cap
    at = E + rows.to(tl.int64) * V + t
    tl.store(at, (tl.load(at, mask=valid).to(tl.float32) * (1 - tanh_y * tanh_y)).to(tl.bfloat16), mask=valid)


@torch.compile(dynamic=False, fullgraph=True)
def capped_losses(z: Tensor, labels: Tensor, c: float) -> Tensor:
    """logsumexp minus the label's entry of the rows of c tanh(z / c), z fp32."""
    z = c * torch.tanh(z / c)
    return torch.logsumexp(z, dim=1) - z.gather(1, labels[:, None]).squeeze(1)


def scaled_exp_loss(x: Tensor, w: Tensor, labels: Tensor, scale: Tensor,
                    cap: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x (M, C), M whole CHUNKs, w (V, C) bf16 -> (per-position loss, dx bf16, dW fp32, per-position logsumexp; 0 at
    IGNORE_INDEX), gradients times scale. One chunk's buffers serve every chunk (one stream orders them)."""
    (M, C), V = x.shape, w.size(0)
    epi = capped_scaled_exp_epi(cap)
    loss, zy = x.new_empty(M, dtype=torch.float32), x.new_empty(M, dtype=torch.float32)
    dx, dw = torch.empty_like(x), x.new_empty((V, C), dtype=torch.float32)
    E, xs, v = x.new_empty((CHUNK, V)), x.new_empty((CHUNK, C)), x.new_empty(CHUNK, dtype=torch.float32)
    sum_exp, k = (x.new_empty((CHUNK, V // TILE_N1), dtype=torch.float32) for _ in range(2))
    strip = x.new_empty((V // 64, CHUNK))  # each chunk's u, contiguous along its rows
    for i, start in enumerate(range(0, M, CHUNK)):
        c = slice(start, start + CHUNK)
        x_c, t_c = x[c], labels[c]
        epi.gemm(x_c, w, E, epi_args={"max_log2": TILE_N1, "sum_exp": sum_exp, "max_log2_out": k, "target": t_c,
                                      "target_logit": zy[c]},
                 tile_M=128, tile_N=TILE_N1, cluster_M=2, cluster_N=1, pingpong=True)
        lce_glue(k, sum_exp, x_c, t_c, zy[c], E, loss[c], strip, v, xs, TILE_N1, ignore_index=IGNORE_INDEX,
                 grad_scale=scale)
        cap_label_fix[(CHUNK // FIX_ROWS,)](E, t_c, zy[c], V, IGNORE_INDEX, 1 / cap, ROWS=FIX_ROWS, num_warps=1)
        _lce_vscale_epi.gemm(transform_a_operand(_lce_dx_scale, E, {"u": strip}, 128, 64), w.mT, dx[c],
                             epi_args={"v": v.unsqueeze(0)}, transform_a=_lce_dx_scale, tile_M=128, tile_N=TILE_N2,
                             tile_K=64, cluster_M=2, cluster_N=1)
        identity_epi.gemm(transform_a_operand(_lce_dw_scale, E.t(), {"u": strip}, 128, 64), xs.mT, dw,
                          epi_args={}, transform_a=_lce_dw_scale, add_to_output=i > 0, tile_M=128, tile_N=TILE_N2,
                          tile_K=64, cluster_M=1, cluster_N=2)
    return loss, dx, dw, torch.where(labels != IGNORE_INDEX, loss + zy, 0.0)


@torch.library.custom_op("sgc::lm_loss", mutates_args=(), device_types="cuda")
def lm_loss(x: Tensor, weight: Tensor, labels: Tensor,
            final_logit_softcapping: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """x (M, C) bf16, M whole CHUNKs, weight (V, C) fp32 -> (mean loss, dx in x's dtype, dW fp32, per-position loss
    and logsumexp (M,) fp32, 0 at IGNORE_INDEX, without gradient); gradients are computed in the forward."""
    scale = 1 / (labels != IGNORE_INDEX).sum().float()
    loss, dx, dw, lse = scaled_exp_loss(x, weight.to(x.dtype), labels, scale, final_logit_softcapping)
    return loss.sum() * scale, dx, dw, loss, lse


@lm_loss.register_fake
def lm_loss_fake(x: Tensor, weight: Tensor, labels: Tensor,
                 final_logit_softcapping: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (x.new_empty((), dtype=torch.float32), x.new_empty(x.shape), weight.new_empty(weight.shape),
            x.new_empty(labels.shape, dtype=torch.float32), x.new_empty(labels.shape, dtype=torch.float32))


def lm_loss_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, float],
                 output: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
    """Only the mean loss is differentiable: a backward through another output raises instead of reading zeros."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(output[1], output[2])


def lm_loss_grad(ctx: FunctionCtx, dloss: Tensor, ddx: Tensor, ddw: Tensor, dlosses: Tensor,
                 dlse: Tensor) -> tuple[Tensor, Tensor, None, None]:
    """Saved gradients times dloss; dx's product in fp32, rounded once to bf16."""
    dx, dw = ctx.saved_tensors
    return (dx.float() * dloss).to(dx.dtype), dw * dloss, None, None


lm_loss.register_autograd(lm_loss_grad, setup_context=lm_loss_save)


def step_shares(labels: list[Tensor]) -> Tensor:
    """Each micro-batch's share of the step's valid labels, fp32: weights that make micro-batch means a step mean."""
    valid = torch.stack([(t != IGNORE_INDEX).sum() for t in labels]).double()
    return (valid / valid.sum()).float()


@torch.library.custom_op("sgc::lm_losses", mutates_args=(), device_types="cuda")
def lm_losses(x: Tensor, weight: Tensor, labels: Tensor, final_logit_softcapping: float) -> Tensor:
    """Per-position loss (M,) fp32, 0 at IGNORE_INDEX labels; no gradient."""
    valid, w = labels != IGNORE_INDEX, weight.to(x.dtype).t()
    rows = torch.where(valid, labels, 0)
    losses = [capped_losses(torch.mm(xc, w, out_dtype=torch.float32), tc, final_logit_softcapping)
              for xc, tc in zip(x.split(CHUNK), rows.split(CHUNK), strict=True)]
    return torch.where(valid, torch.cat(losses), 0.0)


@lm_losses.register_fake
def lm_losses_fake(x: Tensor, weight: Tensor, labels: Tensor, final_logit_softcapping: float) -> Tensor:
    return x.new_empty(labels.shape, dtype=torch.float32)
