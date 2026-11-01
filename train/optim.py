"""
The optimizers (paper §5.1, "Optimizer"). NorMuon (arXiv 2510.05491) trains the attention and MLP weight matrices and
the value-embedding gate weights; fused AdamW trains the token embedding, the output layer (lm_head), the
value-embedding tables and every 1-D parameter. NorMuon's update is rescaled to the root mean square of AdamW's update,
0.2 sqrt(max(fan_out, fan_in)) (Moonlight, arXiv 2502.16982 §2.2), so both optimizers share one peak learning rate and
one weight decay. Before each update the global gradient norm is clipped.
"""
import torch
from torch.optim._muon import _adjust_lr

# AdamW's betas (DeepSeek LLM, arXiv 2401.02954 §2.3)
BETAS = (0.9, 0.95)
# Global gradient-norm clip (DeepSeek LLM, arXiv 2401.02954 §2.3)
MAX_GRAD_NORM = 1.0
# Weight-decay law (paper §5.1, "Weight decay"): the optimal timescale of the weights' exponential moving average,
# as a fraction of the run's steps, is tau* = TAU_C TPP^TAU_E, TPP being tokens per parameter (Power Lines,
# arXiv 2505.13738 App. E.3.1)
TAU_C, TAU_E = 1.084, -0.527
# NorMuon's beta2 (arXiv 2510.05491 §4.1.1); its normalizer eps is that of the reference implementation
# (zichongli5/NorMuon normuon.py normuon_update)
NORMUON_BETA2, NORMUON_EPS = 0.95, 1e-10


def tau_weight_decay(*, lr: float, steps: int, tokens_per_param: float) -> float:
    """The weight decay wd at which the timescale 1 / (lr wd) of the weights' moving average equals
    tau* = TAU_C TPP^TAU_E of the run's steps (Power Lines, arXiv 2505.13738 Finding 3; paper §5.1)."""
    return 1 / (lr * TAU_C * tokens_per_param ** TAU_E * steps)


class NorMuon(torch.optim.Muon):
    """NorMuon (arXiv 2510.05491 Alg. 1 lines 7-9) built on torch.optim.Muon's update. The momentum is Nesterov's (the
    group's `nesterov` option is not read). The Newton-Schulz orthogonalization O runs batched over the matrices of one
    shape, which may round differently from one matrix multiplication per matrix. Each row of O is divided by
    sqrt(v) + eps, v being an exponential moving average of the row's mean square, and the result is rescaled to O's
    Frobenius norm. After each step, update_norms holds, per shape, its parameters and the norms of their updates
    (adjusted learning rate times update), on the device."""

    @torch.no_grad()
    def step(self) -> None:
        self.update_norms = []
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            a, b, c = group["ns_coefficients"]
            shapes: dict[torch.Size, list[torch.Tensor]] = {}
            for p in group["params"]:
                shapes.setdefault(p.shape, []).append(p)
            for shape, params in shapes.items():
                grads = [p.grad for p in params]
                for p in params:
                    if "momentum_buffer" not in self.state[p]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                        self.state[p]["second_momentum_buffer"] = torch.zeros_like(p[:, :1])
                bufs = [self.state[p]["momentum_buffer"] for p in params]
                torch._foreach_lerp_(bufs, grads, 1 - momentum)
                x = torch.stack(torch._foreach_lerp(grads, bufs, momentum)).bfloat16()
                tall = shape[0] > shape[1]
                if tall:
                    x = x.mT
                x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=group["eps"])
                for _ in range(group["ns_steps"]):
                    gram = x @ x.mT
                    x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
                if tall:
                    x = x.mT
                x = x.float()
                second = [self.state[p]["second_momentum_buffer"] for p in params]
                torch._foreach_lerp_(second, list(x.square().mean(-1, keepdim=True).unbind(0)), 1 - NORMUON_BETA2)
                normalized = x / (torch.stack(second).sqrt() + NORMUON_EPS)
                x = normalized * (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
                                  / (torch.linalg.vector_norm(normalized, dim=(-2, -1), keepdim=True) + NORMUON_EPS))
                torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
                adjusted = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                torch._foreach_add_(params, x.unbind(0), alpha=-adjusted)
                self.update_norms.append((params, torch.linalg.vector_norm(x, dim=(-2, -1)) * adjusted))


def optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float) -> tuple[torch.optim.Optimizer, ...]:
    """(AdamW, NorMuon) at the peak learning rate. Decoupled weight decay (arXiv 1711.05101 Alg. 2)
    applies to the matrices, the embedding tables and lm_head, not to 1-D parameters."""
    tables = [model.embed_tokens.weight, model.lm_head.weight, *model.value_embeds.parameters()]
    table_ids = {id(p) for p in tables}
    vectors = [p for p in model.parameters() if p.ndim == 1]
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in table_ids]
    adamw = torch.optim.AdamW([{"params": tables, "weight_decay": weight_decay},
                               {"params": vectors, "weight_decay": 0.0}], lr=lr, betas=BETAS, fused=True)
    return adamw, NorMuon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")


def update(model: torch.nn.Module, opts: tuple[torch.optim.Optimizer, ...],
           accumulated: list[torch.nn.Parameter]) -> torch.Tensor:
    """Clip the accumulated gradients' global norm to MAX_GRAD_NORM, step every optimizer, zero `accumulated`'s grads in
    place and free the rest. Returns the global norm before clipping, on the device (torch.nn.utils.clip_grad_norm_'s
    computation)."""
    params = list(model.parameters())
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in params])))
    torch.nn.utils.clip_grads_with_norm_(params, MAX_GRAD_NORM, norm)
    for opt in opts:
        opt.step()
    in_place = set(accumulated)
    for p in params:
        if p not in in_place:
            p.grad = None
    torch._foreach_zero_([p.grad for p in accumulated])
    return norm
