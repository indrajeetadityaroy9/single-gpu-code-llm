"""
The SFT data (paper §2.2): reasoning traces from Nemotron-Cascade's SFT stages 1 and 2 (code subsets),
Nemotron-SFT-Competitive-Programming-v2 (Python, C++, Exercism) and KodCode-V1-SFT-R1 (its rows whose DeepSeek-R1
solution passed the tests). Each trace is rendered in the chat format (render; paper §5.2.1), filtered, capped at
CAPS traces per prompt, split into train and val by problem (problem), tokenized, and packed whole into sequences of
32,768 tokens by the pretraining packer (corpus/pack.py), one document-masked segment per trace. Each document's
context (the prompt and THINK) length is written as <split>.context; the loader excludes those tokens from the loss
(corpus/loader.py).

python -m corpus.posttrain.build
"""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from corpus import decontaminate, pack
from corpus.layout import TOKEN_DTYPE, split_path
from corpus.parallel import SPAWN
from corpus.posttrain.sft import CLOSE, END_THINK, NAME, OPEN, SFT_STAGE, THINK
from corpus.tokenize import tokenize_split
from corpus.tokenizer import get_tokenizer
from eval.line_endings import lf
from native import corpus_crate
from paths import ARTIFACTS

SOURCE = ARTIFACTS / "posttrain"
# Each group's files (globs under SOURCE) and row format, in the order duplicates are resolved (the first is kept).
# KodCode's `train` split holds the rows whose DeepSeek-R1 solution passed its tests (r1_correctness True; the failed
# ones are the `incorrect` split) and excludes questions similar to benchmarks (cosine similarity > 0.95 to
# HumanEval, MBPP, BigCodeBench or LiveCodeBench, arXiv 2503.02951 §3.2; the `use_with_caution` split)
GROUPS = {
    "cascade_stage1": ("nvidia/Nemotron-Cascade-SFT-Stage-1/code/*.jsonl", "cascade"),
    "cascade_stage2": ("nvidia/Nemotron-Cascade-SFT-Stage-2/code/*.jsonl", "cascade"),
    "cp_python": ("nvidia/Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_python_*.jsonl",
                  "competitive"),
    "cp_cpp": ("nvidia/Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_cpp_*.jsonl",
               "competitive"),
    "cp_exercism": ("nvidia/Nemotron-SFT-Competitive-Programming-v2/data/exercism.jsonl", "exercism"),
    "kodcode": ("KodCode/KodCode-V1-SFT-R1/data/train-*.parquet", "kodcode"),
}
# Traces kept per prompt in each group, drawn at random; None: every trace. The groups with many traces per prompt
# keep at most 2, so that 2 epochs of the packed train sequences stay within 12.5B tokens and the budget goes to
# distinct prompts (paper §2.2); Exercism and KodCode have about one trace per problem
CAPS = {"cascade_stage1": 2, "cascade_stage2": 2, "cp_python": 2, "cp_cpp": 2, "cp_exercism": None, "kodcode": None}
VAL_SHARE = 0.01        # share of problems held out
MIN_REASONING = 20      # minimum reasoning tokens: a shorter trace shows no reasoning before its answer (paper §2.2)
SEED = 0
CHUNK_BYTES = 2**28     # a worker task: one byte range of a source file (the outputs do not depend on it)
SHARD_DOCUMENTS = 4096  # documents per packed shard (one crate pack call)
# Drop reasons, in the order a row is tested
DROPS = ("no_reasoning", "no_answer", "short_reasoning", "over_length", "contaminated", "duplicate", "capped")
# Aider's file listing (CP-v2 Exercism): a filename line, then a fenced block
LISTING = re.compile(r"^(\S+\.\w+)\n```[^\n]*\n(.*?)\n```", re.S | re.M)
# Aider's editing instructions after an Exercism problem statement
AIDER_REQUEST = "\n####\n\nUse the above instructions to modify the supplied files:"
# A markdown heading closing a text (an Exercism reply's "### Solution Code" before its file listing)
TRAILING_HEADING = re.compile(r"\n#{1,6} [^\n]*\s*$")
# The sources' fixed instructions around a problem: CP-v2's header, Nemotron-Cascade's language and format requests
# and its fenced function header (a problem recurs under `def f(x):` and under `class Solution:` with
# `def f(self, x):`)
WRAPPERS = re.compile("|".join((
    r"You are a helpful and harmless assistant\. You should think step-by-step before responding to the instruction "
    r"below\.",
    r"Please use \S+ programming language only\.",
    r"You must use ```\w+ for just the final solution code block with the following format:\s*```\w+\s*\S+ Your code "
    r"here\s*```",
    r"Write (?:Python|C\+\+) code to solve the problem\.",
    r"Solve the problem starting with the provided function header\.\s*Function header:\s*```[\s\S]*?```",
    r"Please place the solution code in the following format:\s*```\w+\s*# Your solution code here\s*```")))

Parsed = tuple[str, str, str] | str  # (prompt, reasoning, answer) or a drop reason


def thinking(prompt: str, reply: str) -> Parsed:
    """The prompt; the reply's text between its leading OPEN and CLOSE, then the rest."""
    head, closed, answer = reply.lstrip().partition(CLOSE)
    if not closed or not head.startswith(OPEN):
        return "no_reasoning"
    return prompt, head.removeprefix(OPEN), answer


def cascade(row: dict) -> Parsed:
    """Nemotron-Cascade: the user message and the reply (thinking)."""
    messages = row["messages"]
    return thinking(messages[0]["content"], messages[1]["content"])


def competitive(row: dict) -> Parsed:
    """CP-v2 Python and C++: the user message; the reply's reasoning_content, then its content."""
    messages = row["messages"]
    return messages[0]["content"], messages[1]["reasoning_content"], messages[1]["content"]


def exercism(row: dict) -> Parsed:
    """CP-v2 Exercism, Aider's 9 messages (system prompt and an example edit first): the stub's file listings
    (message 5 without Aider's preamble) and the problem statement (message 7 without AIDER_REQUEST's instructions);
    the reply's prose before its first listing, less the markdown heading that introduces the code ("### Solution
    Code"), then that listing's code as a Python block."""
    messages = row["messages"]
    stub = LISTING.search(messages[5]["content"])
    reply = messages[8]["content"]
    if (listing := LISTING.search(reply)) is None:
        return "no_answer"
    statement = messages[7]["content"].partition(AIDER_REQUEST)[0]
    return (messages[5]["content"][stub.start():].strip() + "\n\n" + statement.strip(),
            TRAILING_HEADING.sub("", reply[:listing.start()].rstrip()), f"```python\n{listing.group(2)}\n```")


def kodcode(row: dict) -> Parsed:
    """KodCode-V1-SFT-R1: the human turn and the R1 reply (thinking)."""
    turns = row["conversations"]
    return thinking(turns[0]["value"], turns[1]["value"])


FORMATS = {"cascade": cascade, "competitive": competitive, "exercism": exercism, "kodcode": kodcode}


def parse(row: dict, kind: str) -> tuple[str, str, str, bool] | str:
    """A row's (prompt, reasoning, answer), each with LF line endings (eval/line_endings.py lf) and stripped, and
    whether any contained a CR; or the first drop reason found before tokenization."""
    parsed = FORMATS[kind](row)
    if isinstance(parsed, str):
        return parsed
    carriage = any("\r" in text for text in parsed)
    prompt, reasoning, answer = (lf(text).strip() for text in parsed)
    if not reasoning:
        return "no_reasoning"
    if not answer:
        return "no_answer"
    return prompt, reasoning, answer, carriage


def render(prompt: str, reasoning: str, answer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The chat-format document (paper §5.2.1) without its BOS tokens: the (context, reasoning, rest) tokens, each
    encoded separately as ordinary text (the context as inference encodes a prompt)."""
    enc = get_tokenizer().enc
    return tuple(enc.encode_to_numpy(text, disallowed_special=()).astype(TOKEN_DTYPE)
                 for text in (prompt + THINK, reasoning, END_THINK + answer))


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def problem(prompt: str) -> str:
    """A prompt's problem, the unit of the train/val split: the prompt without WRAPPERS, lowercased, its words (runs of
    letters and digits) joined by single spaces, so that one problem under the sources' different wrappers,
    whitespace and markup gives one text."""
    return " ".join(re.findall(r"[^\W_]+", WRAPPERS.sub(" ", prompt).lower()))


def source_rows(path: Path, start: int, end: int) -> Iterator[dict]:
    """A source file's rows starting in bytes [start, end) of a JSON-lines file, or in row groups [start, end) of a
    Parquet file (KodCode's: its conversations)."""
    if path.suffix == ".parquet":
        source = pq.ParquetFile(path)
        for batch in source.iter_batches(row_groups=range(start, end), columns=["conversations"]):
            yield from batch.to_pylist()
        return
    with open(path, "rb") as f:
        f.seek(max(start - 1, 0))
        if start > 0 and f.read(1) != b"\n":  # a row starting before `start` is the previous task's
            f.readline()
        while f.tell() < end and (line := f.readline()):
            yield json.loads(line)


def read_chunk(task: tuple[str, Path, int, int, Path]) -> dict[str, np.ndarray]:
    """The rows of a task's range of a source file (source_rows): each row's drop reason ("reasons", an index into
    DROPS, -1 for a kept candidate) and whether its line endings were normalized ("carriages"); per candidate the
    hashes of its prompt, its problem and its full text, and its context, reasoning and document token counts; the
    candidates' tokens are appended to `part`. Runs in a decontaminate.worker_init worker (decontaminate.GRAMS)."""
    kind, path, start, end, part = task
    reasons, carriages, prompts, problems, keys, contexts, reasonings, lengths = ([] for _ in range(8))
    with open(part, "wb") as out:
        for row in source_rows(path, start, end):
            parsed = parse(row, kind)
            if isinstance(parsed, str):
                reasons.append(DROPS.index(parsed))
                carriages.append(False)
                continue
            prompt, reasoning, answer, carriage = parsed
            carriages.append(carriage)
            context, thought, rest = render(prompt, reasoning, answer)
            length = len(context) + len(thought) + len(rest)
            if len(thought) < MIN_REASONING:
                reasons.append(DROPS.index("short_reasoning"))
            elif length + 2 > SFT_STAGE.seq_len:  # BOS, the document and BOS (one more fits the packer's sequences)
                reasons.append(DROPS.index("over_length"))
            elif decontaminate.GRAMS.matches([f"{prompt}\n{reasoning}\n{answer}"])[0]:
                reasons.append(DROPS.index("contaminated"))
            else:
                reasons.append(-1)
                prompts.append(digest(prompt))
                problems.append(digest(problem(prompt)))
                keys.append(digest(prompt + THINK + reasoning + END_THINK + answer))
                contexts.append(len(context))
                reasonings.append(len(thought))
                lengths.append(length)
                out.write(np.concatenate([context, thought, rest]))
    return {"reasons": np.array(reasons, np.int64), "carriages": np.array(carriages, bool),
            "prompts": np.array(prompts, np.uint64),
            "problems": np.array(problems, np.uint64), "keys": np.array(keys, np.uint64),
            "contexts": np.array(contexts, np.int64), "reasonings": np.array(reasonings, np.int64),
            "lengths": np.array(lengths, np.int64)}


def ranks(values: np.ndarray) -> np.ndarray:
    """Each element's position among the equal values before it."""
    order = np.argsort(values, kind="stable")
    starts = np.r_[True, values[order][1:] != values[order][:-1]]
    rank = np.empty(len(values), np.int64)
    rank[order] = np.arange(len(values)) - np.flatnonzero(starts)[np.cumsum(starts) - 1]
    return rank


def split_stats(lengths: np.ndarray, contexts: np.ndarray, reasonings: np.ndarray, prompts: np.ndarray,
                problems: np.ndarray) -> dict[str, int | list[int]]:
    """Rows, prompts, problems, tokens (BOS, document, BOS), loss tokens (the response and the closing BOS) and the
    10th, 50th and 90th percentiles and maximum of the reasoning tokens per row."""
    return {"rows": len(lengths), "prompts": len(np.unique(prompts)), "problems": len(np.unique(problems)),
            "tokens": int((lengths + 2).sum()), "loss_tokens": int((lengths - contexts + 1).sum()),
            "reasoning_p10_p50_p90_max": np.quantile(reasonings, (0.1, 0.5, 0.9, 1.0)).round().astype(int).tolist()}


def source_tasks() -> list[tuple[int, str, Path, int, int]]:
    """The worker tasks over the sources, in GROUPS order (the order duplicates are resolved in): (group index, row
    format, file, start, end), CHUNK_BYTES byte ranges of a JSON-lines file or a Parquet file's row groups one by
    one."""
    tasks = []
    for g, (pattern, kind) in enumerate(GROUPS.values()):
        for path in sorted(SOURCE.glob(pattern)):
            size = pq.ParquetFile(path).num_row_groups if path.suffix == ".parquet" else path.stat().st_size
            step = 1 if path.suffix == ".parquet" else CHUNK_BYTES
            tasks += [(g, kind, path, start, min(start + step, size)) for start in range(0, size, step)]
    return tasks


def main() -> None:
    tokenizer, rng = get_tokenizer(), np.random.default_rng(SEED)
    directory = split_path(NAME, "train", tokenizer).parent
    directory.mkdir(parents=True)
    parts_dir = directory / "parts"
    parts_dir.mkdir()
    sources = source_tasks()
    owners = [g for g, *_ in sources]
    tasks = [(kind, path, start, end, parts_dir / f"{i}.bin") for i, (_, kind, path, start, end) in enumerate(sources)]
    # the decontamination texts: every benchmark text (benchmark_texts) and the original MBPP (mbpp_texts, the base
    # model's 3-shot evaluation)
    texts = list((decontaminate.benchmark_texts() | decontaminate.mbpp_texts()).values())
    corpus_crate()  # built once here, before the workers load it
    with ProcessPoolExecutor(mp_context=SPAWN, initializer=decontaminate.worker_init, initargs=(texts,)) as pool:
        chunks = list(pool.map(read_chunk, tasks))

    # the candidates, in task order: their group, part file and offset in it
    columns = {k: np.concatenate([c[k] for c in chunks])
               for k in ("prompts", "problems", "keys", "contexts", "reasonings", "lengths")}
    counts = [len(c["lengths"]) for c in chunks]
    group = np.repeat(owners, counts)
    part = np.repeat(np.arange(len(chunks)), counts)
    offset = np.concatenate([np.cumsum(c["lengths"]) - c["lengths"] for c in chunks])
    prompts, problems = columns["prompts"], columns["problems"]
    contexts, reasonings, lengths = columns["contexts"], columns["reasonings"], columns["lengths"]
    duplicate = np.ones(len(lengths), bool)
    duplicate[np.unique(columns["keys"], return_index=True)[1]] = False
    kept, capped = [], np.zeros(len(lengths), bool)
    for g, cap in enumerate(CAPS[name] for name in GROUPS):
        drawn = rng.permutation(np.flatnonzero((group == g) & ~duplicate))
        rank = ranks(prompts[drawn])
        limit = len(drawn) if cap is None else cap
        capped[drawn[rank >= limit]] = True
        kept.append(drawn[rank < limit])
    kept = np.concatenate(kept)
    # val: VAL_SHARE of the distinct problems; train: every other row
    distinct = np.unique(problems[kept])
    held = rng.choice(distinct, size=round(VAL_SHARE * len(distinct)), replace=False)
    val = np.isin(problems[kept], held)
    splits = {"train": rng.permutation(kept[~val]), "val": rng.permutation(kept[val])}

    parts = {i: np.memmap(task[4], dtype=TOKEN_DTYPE, mode="r") for i, task in enumerate(tasks)
             if task[4].stat().st_size}
    for split, rows in splits.items():
        shards = [rows[i:i + SHARD_DOCUMENTS] for i in range(0, len(rows), SHARD_DOCUMENTS)]
        tokenize_split(NAME, split, [1] * len(shards), (
            (np.concatenate([parts[part[i]][offset[i]:offset[i] + lengths[i]] for i in shard]), lengths[shard])
            for shard in shards))
        contexts[rows].tofile(split_path(NAME, split, tokenizer).with_suffix(".context"))
    del parts
    shutil.rmtree(parts_dir)

    stats = {}
    for g, name in enumerate(GROUPS):
        reasons = np.concatenate([c["reasons"] for c, owner in zip(chunks, owners, strict=True) if owner == g])
        mine = group == g
        dropped = dict(zip(DROPS, np.bincount(reasons[reasons >= 0], minlength=len(DROPS)).tolist(), strict=True))
        dropped |= {"duplicate": int((mine & duplicate).sum()), "capped": int((mine & capped).sum())}
        # normalized: the parsed rows (kept or dropped later) whose text contained a CR
        normalized = sum(int(c["carriages"].sum()) for c, owner in zip(chunks, owners, strict=True) if owner == g)
        stats[name] = {"read": len(reasons), "normalized": normalized, "dropped": dropped} | {
            split: split_stats(lengths[r], contexts[r], reasonings[r], prompts[r], problems[r])
            for split, rows in splits.items()
            for r in [rows[group[rows] == g]]}
    stats["all"] = {split: split_stats(lengths[rows], contexts[rows], reasonings[rows], prompts[rows], problems[rows])
                    for split, rows in splits.items()}
    for split in splits:
        packed = pack.write(NAME, split, pack.packed_shards(NAME, split, range(pack.shard_count(NAME, split)),
                                                            SFT_STAGE.seq_len), SFT_STAGE.seq_len)
        # fill: share of input positions in use (each document after its BOS)
        stats["all"][split] |= {"packed_rows": packed,
                                "fill": int((lengths[splits[split]] + 1).sum()) / (packed * SFT_STAGE.seq_len)}
    (directory / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
