"""
The run plan and the learning-rate schedule (paper §5.1, §5.2.1). Pretraining's batch size and peak learning rate come
from DeepSeek LLM's compute scaling laws and its weight decay from the timescale law (train/optim.py); SFT's plan from
the size of its data. Pretraining trains in two stages, the 8,192-token file-level stage, then the 32,768-token
repository-level stage; SFT in one. The schedule is a linear warmup, then linear decay to zero at the last step,
through both stages (arXiv 2502.15938 §4.2).
"""
import math
from dataclasses import dataclass

from corpus.layout import packed_rows
from corpus.mixture import FILE_LEVEL_STAGE, MICRO_BATCH_TOKENS, REPO_LEVEL_SHARE, REPO_LEVEL_STAGE, Stage
from corpus.posttrain.sft import NAME, SFT_STAGE
from corpus.tokenizer import get_tokenizer
from model.modeling import CausalLM, non_embedding_flops_per_token, num_non_embedding_params
from train.optim import BETAS, tau_weight_decay

# Linear warmup over 2 / (1 - beta2) steps, an untuned rule for Adam (arXiv 1910.04209 §4.3 Eq. 14): 40 at beta2 = 0.95
WARMUP_STEPS = round(2 / (1 - BETAS[1]))


@dataclass(frozen=True, kw_only=True)
class RunPlan:
    batch_size: int       # tokens per optimizer step, in whole micro-batches of MICRO_BATCH_TOKENS
    max_steps: int
    repo_level_start: int  # first step of the last stage (the repository-level stage in pretraining)
    learning_rate: float   # peak learning rate of every optimizer group
    weight_decay: float    # of the matrices, embedding tables and lm_head (train/optim.py optimizers)
    last_stage: Stage

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // MICRO_BATCH_TOKENS

    def stage_at(self, step: int) -> Stage:
        """The stage in which step `step` trains (a repository-level stage as in Qwen2.5-Coder, arXiv 2409.12186
        §3.2)."""
        return self.last_stage if step >= self.repo_level_start else FILE_LEVEL_STAGE

    def learning_rate_at(self, step: int) -> float:
        """The learning rate of step `step`: (step + 1) / WARMUP_STEPS of the peak during the warmup, then a linear
        decay that reaches zero at step max_steps."""
        if step < WARMUP_STEPS:
            return self.learning_rate * ((step + 1) / WARMUP_STEPS)
        return self.learning_rate * ((self.max_steps - step) / (self.max_steps - WARMUP_STEPS))


def plan(*, model: CausalLM, tokens_per_param: float, learning_rate_scale: float) -> RunPlan:
    """The pretraining plan of `model` at `tokens_per_param` (paper §5.1, "Batch size and learning rate"): training
    compute C = M D, with M the non-embedding FLOPs per token at the file-level context (DeepSeek LLM's Eq. 2, full
    attention in every layer); batch size B_opt = 0.2920 C^0.3271 tokens rounded to whole micro-batches and peak
    learning rate learning_rate_scale x eta_opt, eta_opt = 0.3118 C^-0.1250 (DeepSeek LLM, arXiv 2401.02954 §3.1
    Eq. 1), the same for both stages; the weight decay of the tau law at that learning rate. The last REPO_LEVEL_SHARE
    of the steps train in the repository-level stage."""
    params = num_non_embedding_params(model)
    tokens = round(tokens_per_param * params)
    compute = non_embedding_flops_per_token(model) * tokens
    batch_size = round(0.2920 * compute ** 0.3271 / MICRO_BATCH_TOKENS) * MICRO_BATCH_TOKENS
    max_steps = tokens // batch_size
    learning_rate = learning_rate_scale * (0.3118 * compute ** -0.1250)
    return RunPlan(batch_size=batch_size, max_steps=max_steps,
                   repo_level_start=max_steps - round(REPO_LEVEL_SHARE * max_steps), learning_rate=learning_rate,
                   weight_decay=tau_weight_decay(lr=learning_rate, steps=max_steps,
                                                 tokens_per_param=max_steps * batch_size / params),
                   last_stage=REPO_LEVEL_STAGE)


def sft_plan(*, epochs: float, batch_size: int, learning_rate: float) -> RunPlan:
    """The SFT plan (paper §5.2.1): SFT_STAGE from the first step, the pretraining run's `batch_size` tokens per step
    (15 micro-batches of 3 rows of 32,768 tokens for the 522M model), as many whole steps as fit in `epochs` passes
    over the packed SFT training rows (train/stages.py stage_batches reads past the first epoch), peak
    `learning_rate`, no weight decay."""
    rows = packed_rows(NAME, "train", get_tokenizer(), SFT_STAGE.seq_len)
    max_steps = math.floor(epochs * rows / (batch_size // MICRO_BATCH_TOKENS * SFT_STAGE.micro_batch_size))
    return RunPlan(batch_size=batch_size, max_steps=max_steps, repo_level_start=0, learning_rate=learning_rate,
                   weight_decay=0.0, last_stage=SFT_STAGE)
