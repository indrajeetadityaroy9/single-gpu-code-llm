"""
Benchmark decontamination of the curated train splits (paper §2.1): a document that shares a 10-gram of
whitespace-separated words with a benchmark prompt or reference solution (benchmark_texts) is removed (Qwen2.5-Coder's
rule, arXiv 2409.12186 §5; DeepSeek-Coder's 10-gram filter, arXiv 2401.14196 §2.4). Validation splits are not
filtered, so every likelihood measurement uses the same set. A shard with a removed document is rewritten without it,
keeping its row groups (for a repository source, whole samples). RECORD lists every removed document. One process per
CPU; runs once, on curate's shards. The original MBPP texts (mbpp_texts) are not among the pretraining corpus's
benchmark texts (the published record lists them); the SFT data's filter adds them (corpus/posttrain/build.py).

python -m corpus.decontaminate
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from huggingface_hub import hf_hub_download
from human_eval_infilling.data import read_problems

from corpus.parallel import SPAWN
from corpus.pretrain.schema import COMPRESSION
from corpus.pretrain.sources import CURATED, SOURCES, curated_shards
from eval.official_infilling import BENCHMARKS
from native import corpus_crate

N = 10
# The removed documents, beside the shards (publish.push uploads it with them)
RECORD = CURATED / "decontamination.json"
GRAMS = None  # a worker's corpus_crate().Grams of the benchmark texts (worker_init)
# The original MBPP (arXiv 2108.07732), evaluated with lm-evaluation-harness's `mbpp_lf` task
# (scripts/eval_pretrained.sh), at a fixed revision
MBPP = {"repo_id": "google-research-datasets/mbpp", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c"}
# MultiPL-E at its generator's pin (multipl_e/completions.py DATASET_REVISION), and HumanEval-X, at fixed revisions
MULTIPL_E = {"repo_id": "nuprl/MultiPL-E", "revision": "3a9e8d226c127392ce81dca29d09f33f0ce3247d"}
HUMANEVAL_X = {"repo_id": "zai-org/humaneval-x", "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198"}


def mbpp_texts() -> dict[str, str]:
    """{mbpp_3shot/<task_id>: text} of the original MBPP's test split (config full): lm-evaluation-harness's prompt
    (tasks/mbpp/mbpp.yaml:7 doc_to_text: the statement and the first three asserts) followed by the reference code."""
    file = "full/test-00000-of-00001.parquet"
    rows = pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **MBPP)).to_pylist()
    return {f"mbpp_3shot/{r['task_id']}": "You are an expert Python programmer, and here is your task: "
            f"{r['text']} Your code should pass these tests:\n\n" + "\n".join(r["test_list"][:3]) + "\n[BEGIN]\n"
            + r["code"] for r in rows}


def benchmark_texts() -> dict[str, str]:
    """{benchmark/task: text} of the scored benchmarks: HumanEval+ and MBPP+ (EvalPlus, arXiv 2305.01210), each its
    prompt (stripped, plus "\\n", as EvalPlus's codegen sends it) and reference solution; MultiPL-E's C++ HumanEval
    and MBPP (arXiv 2208.08227), each its prompt alone (no reference); HumanEval-X C++ (arXiv 2303.17568), each its
    prompt and reference solution; and each HumanEval infilling program (prompt, reference middle and suffix; the
    three infilling benchmarks cut their problems from one program per HumanEval task)."""
    texts = {f"{name}/{task_id}": t["prompt"].strip() + "\n" + t["canonical_solution"]
             for name, load in (("humaneval_plus", get_human_eval_plus), ("mbpp_plus", get_mbpp_plus))
             for task_id, t in load().items()}
    for name in ("humaneval_cpp", "mbpp_cpp"):
        file = f"{name.replace('_', '-')}/test-00000-of-00001.parquet"
        rows = pq.read_table(hf_hub_download(filename=file, repo_type="dataset", **MULTIPL_E)).to_pylist()
        texts |= {f"{name}/{r['name']}": r["prompt"] for r in rows}
    with open(hf_hub_download(filename="data/cpp/data/humaneval.jsonl", repo_type="dataset", **HUMANEVAL_X),
              encoding="utf-8") as f:
        texts |= {f"humaneval_x_cpp/{r['task_id']}": r["prompt"] + r["canonical_solution"] for r in map(json.loads, f)}
    for name in BENCHMARKS:
        for p in read_problems(name).values():
            task = "infilling/" + "/".join(p["task_id"].split("/")[1:3])
            texts[task] = p["prompt"] + p["canonical_solution"] + p["suffix"]
    return texts


def drop_rows(task: tuple[Path, list[int]]) -> None:
    """Rewrite the shard without the given rows (indices into the file), keeping each row group."""
    path, rows = task
    file = pq.ParquetFile(path)
    tmp = path.with_name(f"{path.name}.tmp")
    drop, start = np.asarray(rows), 0
    with pq.ParquetWriter(tmp, file.schema_arrow, compression=COMPRESSION) as writer:
        for g in range(file.num_row_groups):
            table = file.read_row_group(g)
            keep = ~np.isin(np.arange(start, start + table.num_rows), drop)
            start += table.num_rows
            if keep.any():
                writer.write_table(table.filter(keep), row_group_size=int(keep.sum()))
    tmp.replace(path)


def worker_init(texts: list[str]) -> None:
    """Build the worker's GRAMS; one matching thread per worker, since the pool already runs a shard per CPU."""
    global GRAMS
    os.environ["RAYON_NUM_THREADS"] = "1"  # read when the crate builds rayon's pool, at its import
    GRAMS = corpus_crate().Grams(texts, n=N)


def worker_matches(path: Path) -> list[tuple[int, list[int]]]:
    """(row, indices of the matched benchmark texts) of each row of a shard that shares an N-gram with a benchmark
    text, by the worker's GRAMS."""
    found, row = [], 0
    file = pq.ParquetFile(path)
    for g in range(file.num_row_groups):
        contents = file.read_row_group(g, columns=["content"])["content"].to_pylist()
        found += [(row + i, m) for i, m in enumerate(GRAMS.matches(contents)) if m]
        row += len(contents)
    return found


def main() -> None:
    names, texts = zip(*benchmark_texts().items(), strict=True)
    corpus_crate()  # built once here, before the workers load it
    shards = [(source, path) for source in SOURCES for path in curated_shards(source, "train")]
    with ProcessPoolExecutor(mp_context=SPAWN, initializer=worker_init, initargs=(list(texts),)) as pool:
        found = list(pool.map(worker_matches, [path for _, path in shards]))
        dropped = {source: [] for source in SOURCES}
        for (source, path), rows in zip(shards, found, strict=True):
            dropped[source] += [[int(path.stem), row, tests] for row, tests in rows]
        # each removed row: [shard, its row in the shard before decontamination, indices into "benchmarks"]
        RECORD.write_text(json.dumps({"n": N, "benchmarks": names, "dropped": dropped}))
        list(pool.map(drop_rows, [(path, [row for row, _ in rows])
                                  for (_, path), rows in zip(shards, found, strict=True) if rows]))
    for source, rows in dropped.items():
        print(f"{source}: {len(rows):,} train rows dropped", flush=True)


if __name__ == "__main__":
    main()
