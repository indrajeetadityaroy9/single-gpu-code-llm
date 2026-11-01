"""
The RL trainer (paper §5.2.2). Trains the SFT model in the training image with the native model (model/modeling.py,
the fused kernels SFT used), beside rl/sampler.py in the vLLM image on the same GPU. Policies and batches are
exchanged through files (see rl/sampler.py). Training stops after a time budget of BUDGET seconds.

Objective (PPO/GRPO/DAPO lineage). One gradient step per batch, so training is on-policy: the probability ratio is 1
and PPO's clipping never acts (multiple updates per batch "led to rapid entropy collapse around 100 steps",
arXiv:2505.16400 §3.2; Nemotron-Cascade's code RL also makes one, arXiv:2512.13607 §4.1.2). The loss is DAPO's
token-level policy-gradient loss (arXiv:2503.14476 §3.3, Eq. 12):
    -sum_i A_i sum_t log pi(o_i,t) / sum_i |o_i|,
with group-relative advantage A_i = (r_i - group mean reward) / std of the batch's rewards (Lite PPO,
arXiv:2508.08221 §4.1: group mean, batch std). The reward r_i is the sampler's (rl/sampler.py: 1 for an answer that
passes its tests, plus DAPO's soft overlong penalty, so a truncated response scores -1). Masking truncated responses
instead (DAPO's overlong filtering) leaves no signal against length: from the SFT model the truncated share rose from
3.7% to 19.5% in six batches and the pass rate fell from 0.420 to 0.391, where 10-15% truncation "typically
destabilized training" (ScaleRL, arXiv:2510.13786 App. A.15). Dynamic sampling without resampling: only groups whose
rewards differ are trained (equal rewards give zero advantage and no gradient, DAPO §3.2); groups of trivial problems
are unused; no KL penalty (DAPO §2.3); no entropy bonus.

pi is the distribution the sampler draws from at temperature 1.0: the soft-capped logits c tanh(h W / c) of the final
hidden states, with the head in fp32 as vLLM computes it (the export's "head_dtype"). `Head` computes the head and its
loss, with gradients computed in the forward pass, HEAD_CHUNK rows at a time, only at response positions. The
backbone runs over the responses packed into rows of the context length (`pack`: first-fit decreasing, each response
its own attention segment).

Optimizer: AdamW at a constant learning rate LR, betas (0.9, 0.95), no weight decay, gradient clipping at norm 1.0;
eps 1e-15, since an eps near the gradients' scale shrinks the updates (ScaleRL, arXiv:2510.13786 App. A.3). Weights
are fp32; the forward's bf16 GEMM copies are rounded to nearest, and the published policy (inference/export_hf.py
`hf_weights`) has the same rounding, so the sampler samples the policy the trainer differentiates, up to kernel
differences between the two engines.

Per batch k: X/policy-<k> is published (export format; other files copied from the export directory); X/batch-<k>.pkl
is awaited and read; one update. The batch after which the next would overrun BUDGET (elapsed time plus the last
batch's duration) is the last: its policy is written to <out>/final and X/done tells the sampler to stop; otherwise
policy k + 1 is published. <out>/train.jsonl gets one line per batch.

    python -m rl.trainer <checkpoint.pt> <export dir> <exchange dir> <out dir>
"""
import argparse
import itertools
import json
import pickle
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from corpus.tokenizer import get_tokenizer
from inference.export_hf import hf_weights
from model.modeling import accumulated_weights
from model.ops.cross_entropy import CHUNK as HEAD_CHUNK
from train.checkpoint import load_model
from train.optim import BETAS, MAX_GRAD_NORM

LR, EPS = 2e-6, 1e-15
BUDGET = 2 * 3600  # the two-hour training budget, in seconds
# The maximum responses per packed row (attention segments, padding included): cu_seqlens has a fixed MAX_SEGMENTS + 1
# entries, so the compiled forward sees one shape
MAX_SEGMENTS = 256


@torch.compile(dynamic=False, fullgraph=True)
def head_chunk(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
               w: torch.Tensor, cap: float) -> tuple[torch.Tensor, ...]:
    """Loss and gradients of one chunk of the policy head. Inputs: hidden states x (n, C) in bf16, head weight (V, C)
    in fp32, targets, and per-token weights w (advantage / total response tokens). pi is the softmax of the
    soft-capped logits c tanh(x W^T / c), computed in fp32. Returns sum(-w log pi(target)), dx in x's dtype, dW in
    fp32, and per-token log pi(target) and entropy."""
    y = x.float() @ weight.t()
    t = torch.tanh(y / cap)
    z = cap * t
    lse = torch.logsumexp(z, 1)
    logp = z.gather(1, target[:, None]).squeeze(1) - lse
    p = torch.exp(z - lse[:, None])
    entropy = lse - (p * z).sum(1)
    hot = torch.arange(z.size(1), device=z.device) == target[:, None]
    dy = (p - hot.to(p.dtype)) * (1 - t * t) * w[:, None]
    return -(w * logp).sum(), (dy @ weight).to(x.dtype), dy.t() @ x.float(), logp, entropy


class Head(torch.autograd.Function):
    """The policy head as an autograd function: returns sum(-w log pi(target)) over the rows, plus each row's
    log pi(target) and entropy (not differentiated). Gradients are computed in the forward pass, HEAD_CHUNK rows at a
    time, so only one chunk's logits are in memory (as model/ops/cross_entropy.py `lm_loss` does)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, w: torch.Tensor, cap: float):
        n, pad = len(x), -len(x) % HEAD_CHUNK
        x, target, w = F.pad(x, (0, 0, 0, pad)), F.pad(target, (0, pad)), F.pad(w, (0, pad))
        loss, dw = torch.zeros((), device=x.device), torch.zeros_like(weight)
        dx, logp, entropy = [], [], []
        for a in range(0, len(x), HEAD_CHUNK):
            part, d, g, lp, e = head_chunk(x[a:a + HEAD_CHUNK], weight, target[a:a + HEAD_CHUNK],
                                           w[a:a + HEAD_CHUNK], cap)
            loss += part
            dw += g
            dx.append(d)
            logp.append(lp)
            entropy.append(e)
        ctx.save_for_backward(torch.cat(dx)[:n], dw)
        logp, entropy = torch.cat(logp)[:n], torch.cat(entropy)[:n]
        ctx.mark_non_differentiable(logp, entropy)
        return loss, logp, entropy

    @staticmethod
    def backward(ctx, dloss: torch.Tensor, _logp: torch.Tensor, _entropy: torch.Tensor):
        dx, dw = ctx.saved_tensors
        return (dx.float() * dloss).to(dx.dtype), dw * dloss, None, None, None


def pack(lengths: list[int], row: int) -> list[list[int]]:
    """Indices of `lengths` in rows of at most `row` tokens and MAX_SEGMENTS - 1 sequences, first fit decreasing."""
    rows, room = [], []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        for r, free in enumerate(room):
            if lengths[i] <= free and len(rows[r]) < MAX_SEGMENTS - 1:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(row - lengths[i])
    return rows


def packed(members: list[tuple[list[int], int, float]], tokens: int, row: int, pad: int,
           device: torch.device) -> tuple[torch.Tensor, ...]:
    """One packed row of (sequence, prompt length, advantage) members, padded with `pad`. Returns input_ids (1, row),
    cu_seqlens (segment starts), the segment count, the trained positions (response tokens' predecessors), their
    targets and their loss weights (advantage / `tokens`)."""
    ids, bounds = np.full(row, pad, np.int32), []
    index, target, weight = [], [], []
    o = 0
    for s, p, a in members:
        ids[o:o + len(s)] = s
        bounds.append(o)
        index.append(np.arange(o + p - 1, o + len(s) - 1))
        target.append(np.asarray(s[p:], np.int64))
        weight.append(np.full(len(s) - p, a / tokens, np.float32))
        o += len(s)
    if o < row:  # the padding, a segment of its own
        bounds.append(o)
    cu = np.full(MAX_SEGMENTS + 1, row, np.int32)
    cu[:len(bounds)] = bounds
    put = lambda a: torch.from_numpy(a).to(device, non_blocking=True)
    return (put(ids[None]), put(cu), torch.tensor(len(bounds), dtype=torch.int32), put(np.concatenate(index)),
            put(np.concatenate(target)), put(np.concatenate(weight)))


def update(model, hidden, opt, batch: dict) -> dict:
    """One policy-gradient step on a batch; returns its statistics."""
    kept = [(g["samples"], g["prompt"]) for g in batch["groups"] if not g["trivial"]]
    rewards = [np.array([s["reward"] for s in samples], np.float64) for samples, _ in kept]
    flat = np.concatenate(rewards)
    mixed = [(samples, prompt, r) for (samples, prompt), r in zip(kept, rewards, strict=True) if r.min() < r.max()]
    seqs = [(prompt + s["ids"], len(prompt), float(a))
            for samples, prompt, r in mixed for s, a in zip(samples, (r - r.mean()) / flat.std(), strict=True)]
    tokens = sum(len(s) - p for s, p, _ in seqs)
    row, device = model.config.max_position_embeddings, model.lm_head.weight.device
    pad = get_tokenizer().bos_token_id
    loss = logp = entropy = torch.zeros((), device=device, dtype=torch.float64)
    rows = pack([len(s) for s, _, _ in seqs], row)
    for members in rows:
        input_ids, cu_seqlens, segments, index, target, weight = packed([seqs[i] for i in members], tokens, row, pad,
                                                                        device)
        h = hidden(input_ids, cu_seqlens, segments)
        part, lp, e = Head.apply(h[index], model.lm_head.weight, target, weight, model.config.final_logit_softcapping)
        part.backward()
        loss, logp, entropy = loss + part.detach(), logp + lp.sum(), entropy + e.sum()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    opt.step()
    opt.zero_grad(set_to_none=False)
    model.refresh_gemm_weights()
    samples = [s for g in batch["groups"] for s in g["samples"]]
    return {"trivial": sum(g["trivial"] for g in batch["groups"]),
            "pass_rate": sum(s["verdict"] == "pass" for s in samples) / len(samples),
            "verdicts": dict(Counter(s["verdict"] for s in samples)), "reward": float(flat.mean()),
            "mixed_groups": len(mixed), "trained_tokens": tokens, "loss": loss.item(), "grad_norm": norm.item(),
            "logp": logp.item() / tokens, "entropy": entropy.item() / tokens}


def publish(model, source: Path, final: Path) -> None:
    """The policy in the export's format at `final`, written under a temporary name then renamed."""
    tmp = final.with_name(final.name + ".tmp")
    tmp.mkdir()
    save_file(hf_weights(model.state_dict()), tmp / "model.safetensors", metadata={"format": "pt"})
    for f in source.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy(f, tmp / f.name)
    tmp.rename(final)


def main(checkpoint: Path, source: Path, exchange: Path, out: Path) -> None:
    begin = time.time()
    out.mkdir()
    exchange.mkdir()
    model = load_model(checkpoint, device=torch.device("cuda"))
    # fp32 gradients the GEMMs accumulate into (model/ops/linear.py); zeroed, never freed, after each update
    for p in accumulated_weights(model):
        p.grad = torch.zeros_like(p)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=0.0, fused=True)
    hidden = torch.compile(model.hidden, dynamic=False, fullgraph=True)
    publish(model, source, exchange / "policy-0")
    for k in itertools.count():
        start = time.time()
        path = exchange / f"batch-{k}.pkl"
        while not path.exists():
            time.sleep(0.2)
        batch = pickle.loads(path.read_bytes())
        path.unlink()
        shutil.rmtree(exchange / f"policy-{k}")  # loaded by the sampler before it sampled batch k
        stats = update(model, hidden, opt, batch)
        trained = time.time()
        last = trained - begin + (trained - start) > BUDGET
        publish(model, source, out / "final" if last else exchange / f"policy-{k + 1}")
        with open(out / "train.jsonl", "a") as f:
            f.write(json.dumps({"step": k} | stats | {"sampler": batch["stats"], "elapsed": time.time() - begin})
                    + "\n")
        if last:
            (exchange / "done").touch()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("exchange", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.source, args.exchange, args.out)
