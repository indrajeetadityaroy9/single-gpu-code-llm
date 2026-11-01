"""Value-embedding gate and mix (ve_gate.cu; paper §4): v + 2 sigmoid(n W^T) per head x bf16(table[input_ids])
(arXiv 2601.22887 §4.1 Eq. 6). Gate logits are computed in fp32 and the mixed value is rounded once. The backward
adds the table's gradient into table.grad (a leaf that requires grad)."""

import ctypes

import torch
from torch import Tensor

from native import cuda_library, pointer, stream


def ve_forward(n: Tensor, v: Tensor, input_ids: Tensor, table: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """n (B, T, C) bf16, input_ids (B, T) int32, v (B, T, H, HEAD_DIM) bf16 evenly spaced rows, table (V, C) and w
    (H, C) fp32 -> (mixed v contiguous, gate (B*T, H) fp32)."""
    out, gate = v.new_empty(v.shape), v.new_empty((input_ids.numel(), w.size(0)), dtype=torch.float32)
    cuda_library("ve_gate").ve_forward(pointer(n), pointer(v), ctypes.c_longlong(v.stride(1)), pointer(input_ids),
                                       pointer(table), pointer(w), pointer(out), pointer(gate),
                                       ctypes.c_longlong(input_ids.numel()), stream())
    return out, gate


def ve_backward(dv: Tensor, n: Tensor, ids: Tensor, order: Tensor, table: Tensor, w: Tensor,
                gate: Tensor) -> tuple[Tensor, Tensor]:
    """dv laid out as v, ids (B*T,) int32 sorted input_ids, order int64 their rows -> (dn, dw); the table's gradient
    added into table.grad (zeros when None)."""
    if table.grad is None:
        table.grad = torch.zeros_like(table)
    dn, dz = torch.empty_like(n), torch.empty_like(gate, dtype=torch.bfloat16)
    cuda_library("ve_gate").ve_backward(pointer(dv), ctypes.c_longlong(dv.stride(1)), pointer(ids), pointer(order),
                                        pointer(table), pointer(w), pointer(gate), pointer(dn), pointer(table.grad),
                                        pointer(dz), ctypes.c_longlong(ids.numel()), stream())
    return dn, torch.mm(dz.t(), n.view(-1, n.size(-1)), out_dtype=w.dtype)
