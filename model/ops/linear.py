"""The decoder layer's matrix multiplications as torch.library ops on QuACK (Dao-AILab/quack) GEMMs with fixed tile
configs: the fused QKV projection; o_proj with the residual add and the RMSNorm statistics of the result in its
epilogue; and the squared-ReLU MLP, which reads the unnormalized stream because, for RMSNorm without a learned scale,
norm(h) W^T = diag(rstd) h W^T. Backward passes add fp32 dW into the weights' .grad in place (cuBLAS addmm with
beta = 1, as in Megatron-LM's gradient-accumulation fusion); forward-only callers pass None for the grads."""
import torch
from cutlass import BFloat16, Float32
from quack.activation import drelu_sq
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.library import sq_reduce_mod
from quack.epilogue.ops import ColVecLoad, ColVecReduce
from quack.gemm_config import GemmConfig, _get_sm90_configs
from quack.gemm_interface import gemm_norm_act, gemm_tuned
from quack.rms_final_reduce import rms_final_reduce
from torch import Tensor
from torch.autograd.function import FunctionCtx

from model.config import HEAD_DIM

# RMSNorm epsilon, FLT_EPSILON (shared with modeling.py norm and qk_prologue.cu)
EPS = torch.finfo(torch.float32).eps


def sm90(tile_m: int, tile_n: int, cluster_m: int, cluster_n: int, *, swap_ab: bool = False) -> GemmConfig:
    """The point of QuACK's SM90 autotuning space (quack/gemm_config.py _get_sm90_configs) with this tile, cluster and
    operand order; the space sets the rest (pingpong or cooperative by the tile, static persistent scheduler)."""
    (config,) = (c for c in _get_sm90_configs() if (c.tile_m, c.tile_n, c.cluster_m, c.cluster_n, c.swap_ab)
                 == (tile_m, tile_n, cluster_m, cluster_n, swap_ab))
    return config


# Tile config per GEMM at d24 (C = 768) with M = 98,304 rows: the fastest mean of three runs of QuACK's autotuning
# benchmark (GH200 at 700 W). An explicit config bypasses QuACK's tuner.
CONFIGS: dict[str, GemmConfig] = {
    "qkv": sm90(128, 192, 1, 2),
    "qkv_dx": sm90(128, 160, 1, 2, swap_ab=True),
    "o": sm90(128, 192, 1, 2),
    "up": sm90(128, 192, 1, 2),
    "down": sm90(128, 192, 1, 2),
    "down_dx": sm90(128, 192, 2, 1),
    "up_dx": sm90(128, 192, 1, 2),
}


def gemm(A: Tensor, B: Tensor, cfg: GemmConfig, C: Tensor | None = None) -> Tensor:
    """A B (+ C) in A's dtype at config cfg: gemm_tuned's body (QuACK's gemm and gemm_add take no config)."""
    out = A.new_empty(A.shape[0], B.shape[1])
    gemm_tuned.fn(A, B, out, C=C, config=cfg)
    return out


@torch.library.custom_op("sgc::qkv_proj", mutates_args=(), device_types="cuda")
def qkv_proj(x: Tensor, weight: Tensor, gq: Tensor | None, gk: Tensor | None, gv: Tensor | None) -> Tensor:
    """x (..., C), weight (3C, C) stacked bf16 [Wq; Wk; Wv], gq/gk/gv fp32 .grad or None -> (..., 3C) bf16, q, k, v
    the column blocks."""
    x2 = x.reshape(-1, x.shape[-1])
    y = gemm(x2, weight.t(), CONFIGS["qkv"])
    return y.view(*x.shape[:-1], weight.shape[0])


@qkv_proj.register_fake
def qkv_proj_fake(x: Tensor, weight: Tensor, gq: Tensor | None, gk: Tensor | None, gv: Tensor | None) -> Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("sgc::qkv_proj_bwd", mutates_args=("gq", "gk", "gv"), device_types="cuda")
def qkv_proj_bwd(dy: Tensor, x: Tensor, weight: Tensor, gq: Tensor, gk: Tensor, gv: Tensor) -> Tensor:
    """dX like x; each block's dW accumulated into its grad."""
    dy2, x2 = dy.reshape(-1, dy.shape[-1]), x.reshape(-1, x.shape[-1])
    dx = gemm(dy2, weight, CONFIGS["qkv_dx"]).view(x.shape)
    C = gq.shape[0]
    for i, grad in enumerate((gq, gk, gv)):
        torch.addmm(grad, dy2[:, i * C:(i + 1) * C].t(), x2, out_dtype=grad.dtype, out=grad)
    return dx


@qkv_proj_bwd.register_fake
def qkv_proj_bwd_fake(dy: Tensor, x: Tensor, weight: Tensor, gq: Tensor, gk: Tensor, gv: Tensor) -> Tensor:
    return x.new_empty(x.shape)


def qkv_proj_save(ctx: FunctionCtx, inputs: tuple[Tensor, ...], output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def qkv_proj_grad(ctx: FunctionCtx, dy: Tensor) -> tuple[Tensor, None, None, None, None]:
    x, weight, gq, gk, gv = ctx.saved_tensors
    return qkv_proj_bwd(dy, x, weight, gq, gk, gv), None, None, None, None


qkv_proj.register_autograd(qkv_proj_grad, setup_context=qkv_proj_save)


def residual_rms(x: Tensor, weight: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
    """x (M, K), weight (N, K), residual (M, N) -> (h = residual + x W^T, rstd (M,) fp32 of h before rounding)."""
    N = weight.shape[0]
    r = sq_reduce_mod(has_c=True, has_rowvec=False, has_aux=False)(x, weight.t(), residual, config=CONFIGS["o"])
    # gemm_rms's epilogue and finalize (quack/gemm_interface.py _gemm_rms_call, _rms_finalize)
    rstd = rms_final_reduce(r["mColVecReduce"], scale=1 / N, eps=EPS)
    return r["D"], rstd


# tile_n = HEAD_DIM, so each tile's row sums are FlashAttention-4's per-head dpsum; QuACK's tuner does not fix tile_n
DPSUM_CONFIG = sm90(128, HEAD_DIM, 1, 2)


@gemm_epilogue(reduces={"dpsum": ColVecReduce("dpsum", scaled=True)})
def dx_dpsum(acc, c):
    """acc = dO, c = O: D = dO, dpsum = per-tile row sums of bf16(dO) * O, the rowsum(dO * O) term that
    FlashAttention-4's backward needs, computed here instead of in a separate pass."""
    d = acc.to(BFloat16).to(Float32)
    return {"D": d, "dpsum": (d, c)}


def residual_rms_bwd(dh: Tensor, x: Tensor, weight: Tensor, grad: Tensor) -> tuple[Tensor, Tensor]:
    """dh (M, N), x (M, K) FA4's output, grad W's fp32 .grad -> (dx = dh W bf16, dpsum (M, K / HEAD_DIM) fp32);
    dW accumulated into grad."""
    dpsum = dh.new_empty((dh.shape[0], x.shape[1] // HEAD_DIM), dtype=torch.float32)
    dx = dx_dpsum(dh, weight, x, dpsum=dpsum, config=DPSUM_CONFIG)["D"]
    torch.addmm(grad, dh.t(), x, out_dtype=grad.dtype, out=grad)
    return dx, dpsum


@gemm_epilogue(ops={"rstd": ColVecLoad("rstd")}, reduces={"dsum": ColVecReduce("dsum", scaled=True)})
def relu2_dact_rstd(acc, c, rstd):
    """acc = d postact, c = preact: D = dP = rstd dpreact, dsum = row sum of dpreact preact."""
    dpreact, _ = drelu_sq(c, acc)
    return {"D": dpreact * rstd, "dsum": (dpreact, c)}


@gemm_epilogue(ops={"corr": ColVecLoad("corr")})
def norm_bwd_apply(acc, c, h, corr):
    """acc = dP W_up, c = dy, corr = -rstd^2 dsum / C: D = dh."""
    return {"D": acc + c + h * corr}


@torch.library.custom_op("sgc::relu2_mlp", mutates_args=(), device_types="cuda")
def relu2_mlp(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
              g_down: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """h (..., C) un-normed stream, rstd (M,) fp32, g_up/g_down fp32 .grad or None -> (h + MLP(norm(h)), preact,
    postact (M, 4C) saved for the backward); ReLU^2 MLP (arXiv 2109.08668 §3)."""
    h2 = h.reshape(-1, h.shape[-1])
    preact, postact = gemm_norm_act(h2, w_up.t(), rstd=rstd, activation="relu_sq", store_preact=True,
                                    config=CONFIGS["up"])
    y = gemm(postact, w_down.t(), CONFIGS["down"], C=h2)
    return y.view(h.shape), preact, postact


@relu2_mlp.register_fake
def relu2_mlp_fake(h: Tensor, rstd: Tensor, w_up: Tensor, w_down: Tensor, g_up: Tensor | None,
                   g_down: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    act = h.new_empty(h.numel() // h.shape[-1], w_up.shape[0])
    return h.new_empty(h.shape), act, torch.empty_like(act)


@torch.library.custom_op("sgc::relu2_mlp_bwd", mutates_args=("g_up", "g_down"), device_types="cuda")
def relu2_mlp_bwd(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, postact: Tensor, w_up: Tensor, w_down: Tensor,
                  g_up: Tensor, g_down: Tensor) -> Tensor:
    """dh = dy + dP W_up - (rstd^2 sum(dpreact preact) / C) h, dP = rstd dpreact; dW accumulated into g_up, g_down."""
    dy2, h2, C = dy.reshape(-1, dy.shape[-1]), h.reshape(-1, h.shape[-1]), h.shape[-1]
    # The dynamic tile scheduler of QuACK's gemm_dact, whose epilogue this extends (quack/gemm_interface.py gemm_dact)
    r = relu2_dact_rstd(dy2, w_down, preact, rstd=rstd, dynamic_scheduler=True, config=CONFIGS["down_dx"])
    dp = r["D"]
    torch.addmm(g_down, dy2.t(), postact, out_dtype=g_down.dtype, out=g_down)
    corr = rstd.square().mul_(r["dsum"]).mul_(-1 / C)
    dh = norm_bwd_apply(dp, w_up, dy2, h=h2, corr=corr, config=CONFIGS["up_dx"])["D"].view(h.shape)
    torch.addmm(g_up, dp.t(), h2, out_dtype=g_up.dtype, out=g_up)
    return dh


@relu2_mlp_bwd.register_fake
def relu2_mlp_bwd_fake(dy: Tensor, h: Tensor, rstd: Tensor, preact: Tensor, postact: Tensor, w_up: Tensor,
                       w_down: Tensor, g_up: Tensor, g_down: Tensor) -> Tensor:
    return h.new_empty(h.shape)


def relu2_mlp_save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None],
                   output: tuple[Tensor, Tensor, Tensor]) -> None:
    """preact and postact are the backward's operands, not differentiable outputs."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*inputs, *output[1:])


def relu2_mlp_grad(ctx: FunctionCtx, dy: Tensor, dpreact: Tensor,
                   dpostact: Tensor) -> tuple[Tensor, None, None, None, None, None]:
    h, rstd, w_up, w_down, g_up, g_down, preact, postact = ctx.saved_tensors
    return relu2_mlp_bwd(dy, h, rstd, preact, postact, w_up, w_down, g_up, g_down), None, None, None, None, None


relu2_mlp.register_autograd(relu2_mlp_grad, setup_context=relu2_mlp_save)
