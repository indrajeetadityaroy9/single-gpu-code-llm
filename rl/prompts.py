"""
The RL problem pools (paper §2.3). A row of KodCode-V1-SFT-R1's `train` split (each row's DeepSeek-R1 reply passed
its tests) enters the pool of the SFT split (train or val) whose documents start with its SFT context: the prompt and
THINK, encoded as corpus/posttrain/build.py encodes them. One row per distinct context (the first in file order); a
row whose context is in neither split was dropped by build.py (length, contamination, duplicate). Each row keeps
KodCode's tests (a pytest file for the instruct and complete styles, stdin/stdout cases for online_judge) and the R1
answer, used to check the test harness (rl/verify.py --known-good).

A test rewards a correct answer only if it calls what the prompt asks for. For a pytest row, `names` are the names its
test file imports from `solution` (else test_info's function names), and `named` says the prompt writes every one of
them as code (NAMED: `name(` or in backticks); an instruct prompt often omits the function name or gives it only as a
word ("the median"), so a correct answer under another name would fail. `usable` keeps named rows (every online_judge
row) outside the Docs subsets: those test library APIs (pandas, torch, the standard library) whose installed versions
differ from KodCode's, and 45 of 171 Docs rows' R1 answers failed here in a 2,000-row known-good sample (no record).
The pools are written beside the SFT corpus as onpolicy-<split>.parquet; `sample` draws prompts stratified by style
and difficulty.

    python -m rl.prompts
"""
import ast
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus.layout import TOKEN_DTYPE, split_path
from corpus.parallel import SPAWN
from corpus.posttrain.build import GROUPS, SOURCE, parse
from corpus.posttrain.sft import NAME, THINK
from corpus.tokenizer import get_tokenizer

SPLITS = ("train", "val")
COLUMNS = ("question_id", "style", "subset", "gpt_difficulty", "test", "test_info", "conversations")
NAMED = r"(?<![\w.]){name}\s*\(|`{name}`"


def names(test: str, info: list[dict] | None) -> list[str]:
    """The names a pytest file imports from solution (every KodCode pytest file parses), else test_info's function
    names."""
    found = [a.name for node in ast.walk(ast.parse(test)) if isinstance(node, ast.ImportFrom)
             and node.module == "solution" for a in node.names if a.name != "*"]
    return sorted(set(found)) or sorted({f["function_name"] for f in info or [] if f.get("function_name")})


def key(tokens) -> bytes:
    return hashlib.blake2b(np.asarray(tokens, TOKEN_DTYPE).tobytes(), digest_size=16).digest()


def split_contexts(split: str) -> set[bytes]:
    """The keys of the split's document contexts (<split>.context tokens at the start of each document)."""
    path = split_path(NAME, split, get_tokenizer())
    tokens = np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
    offsets = np.fromfile(path.with_suffix(".idx"), np.int64)
    contexts = np.fromfile(path.with_suffix(".context"), np.int64)
    return {key(tokens[a:a + c]) for a, c in zip(offsets[:-1].tolist(), contexts.tolist(), strict=True)}


def pool_path(split: str) -> Path:
    return split_path(NAME, split, get_tokenizer()).with_name(f"onpolicy-{split}.parquet")


def read_file(path: Path) -> list[tuple[bytes, dict]]:
    """A source file's parsed rows (build.py parse): (context key, the pool row)."""
    tok = get_tokenizer()
    out = []
    for batch in pq.ParquetFile(path).iter_batches(columns=list(COLUMNS), batch_size=4096):
        for row in batch.to_pylist():
            parsed = parse(row, "kodcode")
            if isinstance(parsed, str):
                continue
            prompt, _, answer, _ = parsed
            context = tok.enc.encode_ordinary(prompt + THINK)
            stdio = row["style"] == "online_judge"
            entry = [] if stdio else names(row["test"], row["test_info"])
            out.append((key(context), {
                "id": f"{row['question_id']}/{row['style']}", "style": row["style"], "subset": row["subset"],
                "difficulty": row["gpt_difficulty"], "prompt": prompt, "context_tokens": len(context),
                "kind": "stdio" if stdio else "pytest", "test": row["test"], "names": entry,
                "named": all(re.search(NAMED.format(name=re.escape(n)), prompt) for n in entry),
                "r1_answer": answer}))
    return out


def build() -> None:
    splits = {split: split_contexts(split) for split in SPLITS}
    files = sorted(SOURCE.glob(GROUPS["kodcode"][0]))
    with ProcessPoolExecutor(max_workers=len(files), mp_context=SPAWN) as pool:
        parsed = list(pool.map(read_file, files))
    rows, seen, ids = {split: [] for split in SPLITS}, set(), Counter()
    for k, row in (item for chunk in parsed for item in chunk):
        split = next((s for s in SPLITS if k in splits[s]), None)
        if split is None or k in seen:
            continue
        seen.add(k)
        ids[row["id"]] += 1
        if ids[row["id"]] > 1:  # one question and style under two prompt texts
            row["id"] += f"#{ids[row['id']]}"
        rows[split].append(row)
    for split, rs in rows.items():
        pq.write_table(pa.Table.from_pylist(rs), pool_path(split))
        print(split, len(rs), json.dumps(sorted(Counter(f"{r['style']}/{r['difficulty']}/{r['named']}"
                                                      for r in rs).items())), flush=True)


def read_pool(split: str) -> list[dict]:
    return pq.read_table(pool_path(split)).to_pylist()


def usable(row: dict) -> bool:
    """Whether a row's tests can tell a correct answer (see the module docstring): the prompt names every tested name,
    the row is outside the Docs subsets, and a pytest row has at least one tested name (with none, `named` is
    trivially true and the stub would define nothing)."""
    return row["named"] and (row["kind"] == "stdio" or bool(row["names"])) and not row["subset"].startswith("Docs")


def question(row: dict) -> str:
    """The KodCode question a row renders: its question id without the style suffix (_I instruct, _C complete, _OJ
    online judge), so the styles of one question share it."""
    return re.sub(r"_(I|C|OJ)$", "", row["id"].split("/")[0])


def held_out(val: list[dict], train: list[dict]) -> list[dict]:
    """The val rows whose question and test file both differ from those of every train-pool row. Needed because SFT
    splits by problem statement, and KodCode words one question differently in each style: another style of a train
    question, or a question with a train row's tests, can sit in val (paper §2.3)."""
    questions, tests = {question(r) for r in train}, {r["test"] for r in train}
    return [r for r in val if question(r) not in questions and r["test"] not in tests]


def sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """n of `pool`'s usable rows without replacement, each (style, difficulty) stratum's share proportional to its
    usable size (largest remainder), drawn by default_rng(seed); in id order."""
    pool = [row for row in pool if usable(row)]
    strata: dict[tuple[str, str], list[dict]] = {}
    for row in sorted(pool, key=lambda r: r["id"]):
        strata.setdefault((row["style"], row["difficulty"]), []).append(row)
    keys = sorted(strata)
    exact = np.array([n * len(strata[s]) / len(pool) for s in keys])
    quota = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - quota), kind="stable")[:n - quota.sum()]:
        quota[i] += 1
    rng = np.random.default_rng(seed)
    picked = [strata[s][i] for s, q in zip(keys, quota, strict=True)
              for i in rng.choice(len(strata[s]), size=q, replace=False)]
    return sorted(picked, key=lambda r: r["id"])


def context_ids(row: dict) -> list[int]:
    """The prompt's ids as the SFT trained on them and inference sends them: BOS, then the context."""
    tok = get_tokenizer()
    return [tok.bos_token_id, *tok.enc.encode_ordinary(row["prompt"] + THINK)]


if __name__ == "__main__":
    build()
