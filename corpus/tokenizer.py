"""
The corpus's tokenizer (paper §3.1): a byte-level BPE vocabulary trained by SA-BPE (train_tokenizer), stored under
CORPUS/tokenizer (tokenizer.tiktoken, tokenizer.pattern) and encoded by tiktoken. The tokenized and packed data are
keyed by its data key. Also builds fill-in-the-middle prompts (paper §3.3).
"""

import base64
import functools
import hashlib
import json

import numpy as np
import tiktoken

from paths import CORPUS

TOKENIZER = CORPUS / "tokenizer"

BOS = "<|bos|>"
# FIM special tokens named as in deepseek-ai/deepseek-coder-6.7b-base tokenizer.json (arXiv 2401.14196 §3.1.2)
FIM_BEGIN, FIM_HOLE, FIM_END = ("<\uff5cfim\u2581begin\uff5c>", "<\uff5cfim\u2581hole\uff5c>",
                                "<\uff5cfim\u2581end\uff5c>")
FIM_TOKENS = (FIM_BEGIN, FIM_HOLE, FIM_END)
SPECIAL_TOKENS = (BOS, *FIM_TOKENS)  # ids after the mergeable tokens, in this order

# The pre-tokenizer: cl100k's regular expression (as tiktoken 0.6.0 writes it, digit runs of one or two) without the
# [\r\n]* after a punctuation run, so no token joins punctuation to a line break (arXiv 2608.18062 App. F.3, fn. 16;
# paper §3.1)
SPLIT_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")


def tiktoken_file(mergeable: list[bytes]) -> str:
    """tiktoken's file format: each mergeable token in base64 and its rank."""
    return "".join(f"{base64.b64encode(token).decode()} {rank}\n" for rank, token in enumerate(mergeable))


class Tokenizer:
    """A vocabulary: its mergeable tokens' bytes in rank order, then SPECIAL_TOKENS, encoded by tiktoken."""

    def __init__(self, mergeable: list[bytes], pattern: str) -> None:
        self.pattern, self.mergeable = pattern, mergeable
        self.bos_token_id = len(mergeable)
        self.special_tokens = {token: len(mergeable) + i for i, token in enumerate(SPECIAL_TOKENS)}
        # a special token's bytes are its text; its length in document bytes is 0
        self.token_bytes = [*mergeable, *(token.encode() for token in SPECIAL_TOKENS)]
        self.token_lengths = np.array([len(b) for b in mergeable] + [0] * len(SPECIAL_TOKENS), dtype=np.int64)
        self.vocab_size = len(self.token_bytes)
        # SHA-256 of [pattern, {BOS: id}, tiktoken file]; the FIM tokens after BOS are not part of it
        identity = json.dumps([pattern, {BOS: self.bos_token_id}, tiktoken_file(mergeable)])
        self.data_key = hashlib.sha256(identity.encode()).hexdigest()
        self.enc = tiktoken.Encoding(self.data_key, pat_str=pattern, special_tokens=self.special_tokens,
                                     mergeable_ranks={token: rank for rank, token in enumerate(mergeable)})


@functools.cache
def get_tokenizer() -> Tokenizer:
    """The corpus's tokenizer, from TOKENIZER."""
    with open(TOKENIZER / "tokenizer.tiktoken", "rb") as f:
        mergeable = [base64.b64decode(line.split()[0]) for line in f]
    with open(TOKENIZER / "tokenizer.pattern", newline="") as f:
        return Tokenizer(mergeable, f.read())


def fim_prompt(tokenizer: Tokenizer, prefix: str, suffix: str) -> list[int]:
    """The token ids of a fill-in-the-middle prompt in PSM order (arXiv 2207.14255 §3, App. D), prefix and suffix
    encoded separately: BOS <fim_begin> prefix <fim_hole> suffix <fim_end>. The model continues with the middle and
    ends it with BOS."""
    special = tokenizer.special_tokens
    p, s = tokenizer.enc.encode_ordinary(prefix), tokenizer.enc.encode_ordinary(suffix)
    return [tokenizer.bos_token_id, special[FIM_BEGIN], *p, special[FIM_HOLE], *s, special[FIM_END]]
