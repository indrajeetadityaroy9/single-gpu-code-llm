"""Stops looping responses inside `vllm serve` for scripts/eval_posttrained.sh (paper §6.1, "Finished answers").

A loop is reasoning that keeps repeating the same text. The rule (eval/loop.py) is the one the paper states: from
REPEAT_TAIL (4,096) generated tokens on, every CHUNK (2,048) tokens, a response that has not written "</think>" is
stopped if at least LOOP (99%) of its last 4,096 tokens lie inside a REPEAT_WINDOW (32)-token sequence that occurs
more than once within them. The termination probe (eval/official_termination.py `generate`) applies the same rule.

This module implements the rule as a logits processor for vLLM 0.32's model runner V2 (the default). A stopped
request gets the end-of-sequence token (BOS) as its next token. The server then reports finish reason "stop", and the
reasoning parser gives an empty answer because there is no "</think>" (eval/official_reasoning_parser.py). Every
scorer fails an empty answer, exactly as it would fail the same loop left to run to max_tokens. In the probe's terms,
the response is "unclosed_loop_stopped" at the same length.

Clients identify a stopped response from its token ids (eval/loop.py `stopped`).

Cost per step: a host-side check of each request's output length. Only a request at a check copies its tokens back
from the GPU.

    vllm serve ... --logits-processors eval.official_loop_stop:LoopStopLogitsProcessor
"""

from vllm.v1.worker.gpu.sample.logits_processor.interface import (LogitsContext, LogitsProcessor,
                                                                  LogitsProcRequestState)

from corpus.tokenizer import get_tokenizer
from eval.loop import CHUNK, REPEAT_TAIL, looping


class LoopStopLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states: LogitsProcRequestState):
        tok = get_tokenizer()
        self.bos, self.decode = tok.bos_token_id, tok.enc.decode
        self.req_states = req_states

    def apply(self, logits, ctx: LogitsContext):
        slots = ctx.idx_mapping_np
        seq_lens = ctx.seq_lens_upper_bound_np[:len(slots)]
        generated = seq_lens - self.req_states.prompt_len.np[slots]
        # a check: n generated tokens at a check length, and not a chunk inside a refill after preemption
        refilled = seq_lens >= self.req_states.prefill_len.np[slots]
        for i in ((generated >= REPEAT_TAIL) & (generated % CHUNK == 0) & refilled).nonzero()[0].tolist():
            slot, n = int(slots[i]), int(generated[i])
            if looping(self.output(slot, n, ctx), self.decode) is None:
                continue
            rows = (ctx.expanded_idx_mapping == slot).nonzero().squeeze(1)
            logits[rows] = float("-inf")
            logits[rows, self.bos] = 0.0
        return logits

    def output(self, slot: int, n: int, ctx: LogitsContext) -> list[int]:
        """The request's n generated tokens: committed ones from the token history (valid up to total_len), the last
        from this step's input (its row's input id) when not yet committed."""
        start = int(self.req_states.prompt_len.np[slot])
        end = start + n
        committed = min(int(self.req_states.total_len.gpu[slot]), end)
        ids = self.req_states.all_token_ids.gpu[slot, start:committed].tolist()
        if committed < end:
            rows = (ctx.expanded_idx_mapping == slot).nonzero().squeeze(1)
            ids += ctx.input_ids[rows][:end - committed].tolist()
        return ids
