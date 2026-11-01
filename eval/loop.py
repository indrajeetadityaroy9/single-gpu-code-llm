"""The loop rule for post-trained models' reasoning (paper §6.1, "Finished answers"), shared by the termination probe
(eval/official_termination.py `generate`) and the server-side loop stop (eval/official_loop_stop.py). It imports only
numpy and the chat format's tags (corpus/posttrain/sft.py, itself numpy only), because every `vllm serve` process
imports it.

A loop is reasoning that keeps repeating the same text. From REPEAT_TAIL generated tokens on, at a check every CHUNK
tokens, a response is stopped when its output holds no END ("</think>") and its last REPEAT_TAIL tokens have
`repetition` >= LOOP.
"""

from collections import Counter

import numpy as np

from corpus.posttrain.sft import CLOSE as END  # the reasoning's close: the reasoning parser's split

CHUNK = 2048  # generated tokens between checks (the probe's rounds)
REPEAT_TAIL, REPEAT_WINDOW, LOOP = 4096, 32, 0.99


def covered(ids: list[int]) -> np.ndarray:
    """Per token of ids, whether it lies inside a REPEAT_WINDOW-token window that recurs within ids."""
    windows = [tuple(ids[i:i + REPEAT_WINDOW]) for i in range(len(ids) - REPEAT_WINDOW + 1)]
    counts = Counter(windows)
    out = np.zeros(len(ids), bool)
    for i, w in enumerate(windows):
        if counts[w] > 1:
            out[i:i + REPEAT_WINDOW] = True
    return out


def repetition(ids: list[int]) -> float:
    """The share of the last REPEAT_TAIL tokens inside a REPEAT_WINDOW-token window that recurs within them."""
    tail = ids[-REPEAT_TAIL:]
    return float(covered(tail).mean()) if len(tail) else 0.0


def looping(output: list[int], decode) -> float | None:
    """The `repetition` of an output that the rule stops at a check (len(output) a positive multiple of CHUNK, at least
    REPEAT_TAIL), else None; decode: token ids -> text."""
    n = len(output)
    if n < REPEAT_TAIL or n % CHUNK:
        return None
    r = repetition(output)
    return r if r >= LOOP and END not in decode(output) else None


def stopped(ids: list[int], bos: int, decode) -> bool:
    """Whether a served response (its generated ids, the closing BOS included) was ended by the loop stop: it ends
    with the BOS forced at a check the rule stops (a BOS the model emits there itself would have been forced)."""
    return ids[-1:] == [bos] and looping(ids[:-1], decode) is not None
