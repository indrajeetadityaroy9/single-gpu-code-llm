"""Fused QK-norm and RoPE (qk_prologue.cu; paper §4): per-head RMSNorm of q and k, a learned fp32 per-dimension scale
(QK-norm as in arXiv 2505.09388 §2) and RoPE over the full head (arXiv 2104.09864 §3.2.2), in one kernel; math in
fp32, one bf16 rounding per output. q, k (B, T, H, HEAD_DIM) bf16 are views into the fused QKV projection output
(token stride 3C); scales (HEAD_DIM,) and cos/sin tables are fp32."""

import ctypes

import torch
from torch import Tensor

from native import cuda_library, pointer, stream


def qk_forward(q: Tensor, k: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, cos: Tensor,
               sin: Tensor) -> tuple[Tensor, Tensor]:
    """(normed and rotated q, k), contiguous."""
    B, T, H = q.shape[:3]
    q_out, k_out = q.new_empty(q.shape), k.new_empty(k.shape)
    cuda_library("qk_prologue").qkp_forward(
        pointer(q), pointer(k), pointer(q_norm_weight), pointer(k_norm_weight), pointer(q_out), pointer(k_out),
        pointer(cos), pointer(sin), ctypes.c_longlong(B * T * H), H, T, stream())
    return q_out, k_out


def qk_backward(q: Tensor, k: Tensor, q_norm_weight: Tensor, k_norm_weight: Tensor, dq_out: Tensor, dk_out: Tensor,
                cos: Tensor, sin: Tensor, dq: Tensor, dk: Tensor) -> tuple[Tensor, Tensor]:
    """Write q's and k's gradients into dq, dk; return the gains' gradients, summed in a fixed order (bitwise)."""
    B, T, H, D = q.shape
    dq_norm_weight, dk_norm_weight = torch.empty_like(q_norm_weight), torch.empty_like(k_norm_weight)
    lib = cuda_library("qk_prologue")
    part = q_norm_weight.new_empty((2 * lib.qkp_backward_blocks(), D))  # per-block partial weight gradients
    lib.qkp_backward(
        pointer(q), pointer(k), pointer(q_norm_weight), pointer(k_norm_weight), pointer(dq_out), pointer(dk_out),
        pointer(dq), pointer(dk), pointer(dq_norm_weight), pointer(dk_norm_weight), pointer(part), pointer(cos),
        pointer(sin), ctypes.c_longlong(B * T * H), H, T, stream())
    return dq_norm_weight, dk_norm_weight
