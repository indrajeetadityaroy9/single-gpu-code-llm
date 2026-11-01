"""
Per-step training diagnostics behind the stability results of paper §5.1 (log-sum-exp of the soft-capped logits,
QK-norm scales, residual scalars, update sizes). They are computed on the GPU and copied to the host without blocking,
so they are read one step late, together with the loss and the gradient norm.

Each step's train record gets the largest and the mean square log-sum-exp of the output logits (the quantity PaLM's
z-loss penalizes, arXiv 2204.02311 §5). A float32 side file (RUNS/<run>/STEP_FILE, columns named in COLUMNS_FILE)
gets, every step, each QK-norm scale vector's largest entry and the residual scalars resid_lambdas and x0_lambdas.
Every UPDATE_STATS_EVERY steps, UPDATE_FILE gets each tensor's update RMS (without weight decay).
"""

import json

import numpy as np
import torch
from torch import Tensor

from corpus.loader import IGNORE_INDEX
from model.modeling import CausalLM
from paths import RUNS

# measuring every 10th step costs less than the throughput noise on the GH200
UPDATE_STATS_EVERY = 10
# The side files under RUNS/<run>/: .npy float32 arrays; row s holds step s (STEP_FILE) or step s x UPDATE_STATS_EVERY
# (UPDATE_FILE). COLUMNS_FILE names their columns and describes the tensors.
STEP_FILE, UPDATE_FILE, COLUMNS_FILE = "diagnostics-step.npy", "diagnostics-update.npy", "diagnostics.json"
# end_step's leading scalars: loss, grad_norm, logsumexp max, square sum and count
SCALARS = 5


class Diagnostics:
    def __init__(self, model: CausalLM, adamw: torch.optim.AdamW, muon: torch.optim.Muon, *, run_name: str,
                 max_steps: int) -> None:
        self.adamw, self.muon = adamw, muon
        named = list(model.named_parameters())
        self.params = [p for _, p in named]
        gains = [(name, p) for name, p in named if name.endswith(("q_norm_weight", "k_norm_weight"))]
        self.gains = [p for _, p in gains]
        self.lambdas = (model.resid_lambdas, model.x0_lambdas)
        groups = {p: f"{name}[{i}]" for name, opt in (("adamw", adamw), ("muon", muon))
                  for i, group in enumerate(opt.param_groups) for p in group["params"]}
        layers = range(model.config.num_hidden_layers)
        step_columns = ([f"gain_max/{name}" for name, _ in gains] + [f"resid_lambdas/{i}" for i in layers]
                        + [f"x0_lambdas/{i}" for i in layers])
        update_columns = [f"update_rms/{name}" for name, _ in named]
        directory = RUNS / run_name
        (directory / COLUMNS_FILE).write_text(json.dumps({
            "update_stats_every": UPDATE_STATS_EVERY,
            # layer: the decoder layer of layers.<i>.* and value_embeds.<i>.*, null for the model-wide tensors
            "tensors": [{"name": name, "shape": list(p.shape), "group": groups[p],
                         "layer": int(name.split(".")[1]) if name.startswith(("layers.", "value_embeds.")) else None}
                        for name, p in named],
            STEP_FILE: step_columns, UPDATE_FILE: update_columns}, indent=1), encoding="utf-8")
        self.step_rows = np.lib.format.open_memmap(directory / STEP_FILE, mode="w+", dtype=np.float32,
                                                   shape=(max_steps, len(step_columns)))
        self.update_rows = np.lib.format.open_memmap(directory / UPDATE_FILE, mode="w+", dtype=np.float32,
                                                     shape=(-(-max_steps // UPDATE_STATS_EVERY), len(update_columns)))
        # two pinned host buffers of a step's values, alternating: a step's are read while the next step runs
        size = SCALARS + len(step_columns) + len(update_columns)
        self.host = [torch.empty(size, dtype=torch.float64, pin_memory=True) for _ in range(2)]

    def begin_step(self, step: int) -> None:
        self.lse_max, self.lse_square, self.lse_count = [], [], []
        self.measured = step % UPDATE_STATS_EVERY == 0

    @torch.no_grad()
    def micro_batch(self, labels: Tensor, lse: Tensor) -> None:
        """One micro-batch's logsumexp statistics over its valid labels. No autograd: a graph copied into the pinned
        buffers would chain every step's onto the last, never freed."""
        valid = labels.flatten() != IGNORE_INDEX
        self.lse_max.append(torch.where(valid, lse, -torch.inf).amax())
        self.lse_square.append((lse.square() * valid).sum())
        self.lse_count.append(valid.sum(dtype=torch.float32))

    def update_rms(self) -> Tensor:
        """Each tensor's update RMS without weight decay, in model order: AdamW's lr m̂ / (√v̂ + eps) from its state,
        NorMuon's from NorMuon.update_norms."""
        rms = {}
        for group in self.adamw.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                state = self.adamw.state[p]
                step_count = state["step"]
                rms[p] = (state["exp_avg"] / (1 - beta1**step_count) / (
                    state["exp_avg_sq"].sqrt() / (1 - beta2**step_count).sqrt() + group["eps"])
                ).square().mean().sqrt() * group["lr"]
        for params, norms in self.muon.update_norms:
            for p, norm in zip(params, norms.unbind(0), strict=True):
                rms[p] = norm / p.numel() ** 0.5
        return torch.stack([rms[p] for p in self.params])

    @torch.no_grad()
    def end_step(self, step: int, loss: Tensor, grad_norm: Tensor) -> tuple[Tensor, bool]:
        """After the optimizers stepped: the step's values copied to a pinned host buffer without blocking, and whether
        they include the update RMS."""
        parts = [torch.stack([loss, grad_norm, torch.stack(self.lse_max).amax(), torch.stack(self.lse_square).sum(),
                              torch.stack(self.lse_count).sum()]),
                 torch.stack(torch._foreach_norm(self.gains, float("inf"))), *self.lambdas]
        if self.measured:
            parts.append(self.update_rms())
        values = torch.cat(parts).double()
        host = self.host[step % 2][:values.numel()]
        host.copy_(values, non_blocking=True)
        return host, self.measured

    def record(self, step: int, values: np.ndarray, measured: bool) -> dict:
        """Writes a step's side-file rows from its host values (after its end event) and returns the diagnostics for
        its train record."""
        end = SCALARS + self.step_rows.shape[1]
        self.step_rows[step] = values[SCALARS:end]
        if measured:
            self.update_rows[step // UPDATE_STATS_EVERY] = values[end:]
        return {"max_logsumexp": float(values[2]), "mean_square_logsumexp": float(values[3] / values[4])}

    def flush(self) -> None:
        self.step_rows.flush()
        self.update_rows.flush()
