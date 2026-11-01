"""
Publication of the pretraining corpus (paper §2.1 "Published corpus"): uploads the curated shards, the
decontamination record and the tokenizer to the Hugging Face dataset DATASET, and downloads them back. push replaces
the dataset's files in one commit; pull downloads the commit REVISION.

python -m corpus.pretrain.publish push | pull | pull_tokenizer
"""

import argparse

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download

from corpus.decontaminate import RECORD
from corpus.pretrain.schema import REPOSITORY_FILE, SAMPLE_FILE
from corpus.pretrain.sources import CURATED, SOURCES, curated_dir, footer_rows
from corpus.tokenizer import TOKENIZER
from paths import CORPUS

DATASET = "iaroy/sgc-data"
# The dataset commit that pull downloads: push's decontaminated corpus, the version pretraining read
REVISION = "dbe57a81ba0314a501381da2331e598bc993be4c"

# Bytes per preupload batch: an LFS upload URL expires 900 s after its batch call, so 4 GiB needs at least 4.6 MiB/s
PREUPLOAD = 4 * 1024**3
CONTENTS = {
    "stack_edu_python": "GitHub Python files filtered for educational value",
    "stack_edu_cpp": "GitHub C++ files filtered for educational value",
    "ultradata_stack_v3_l2_python": "GitHub Python files, quality-filtered",
    "ultradata_stack_v3_l2_cpp": "GitHub C++ files, quality-filtered",
    "ultradata_code_l2_python": "GitHub Python files, quality-filtered",
    "ultradata_code_l2_cpp": "GitHub C++ files, quality-filtered",
    "ultradata_code_l3_python": "synthetic Python exercises: a task, then its solution",
    "ultradata_code_l3_cpp": "synthetic C++ exercises: a task, then its solution",
    "nemotron_rewriting_python": "GitHub Python files rewritten by an LLM (Qwen3-32B)",
    "nemotron_transpilation_cpp": "GitHub Python files transpiled into C++ by an LLM (Qwen3-32B)",
    "codexglue_python": "Python functions with their docstrings (CodeSearchNet)",
    "ultradata_math_l3_qa": "synthetic mathematics: questions and answers",
    "ultradata_math_l3_conversation": "synthetic mathematics: conversations",
    "ultradata_math_l3_multi_style": "synthetic mathematics: varied styles",
    "ultradata_math_l3_textbook_exercise": "synthetic mathematics: textbook exercises",
    "codeforces_cots": "Codeforces problems with finished DeepSeek-R1 reasoning traces and Python or C++ solutions",
    "kaggle_notebooks": "Kaggle notebooks in Markdown: title, dataset description, the notebook as a Python script",
    "climbmix": "English web text",
}
COLUMNS = {
    "content": "the document's text",
    "repo_name": "its GitHub repository, `owner/name`",
    "relative_path": "the file's path in its repository",
    "sample": "within a shard, files sharing a number import one another; they are consecutive, in dependency order",
    "id": "the SHA-1 of `content`",
    "language": "the code's language, `Python` or `C++`",
    "problem_id": "the Codeforces problem",
}


def card() -> str:
    """The dataset card, with row counts from the curated shards' Parquet footers. The Hub's dataset viewer is
    disabled because its Parquet converter would merge the sources' different schemas."""
    rows = {source: {split: footer_rows(curated_dir(source, split).iterdir()) for split in ("train", "val")}
            for source in SOURCES}
    folders = []
    for source, splits in rows.items():
        repo = SOURCES[source].repo
        category = SOURCES[source].category
        columns = (SAMPLE_FILE if category == REPOSITORY_FILE else category.schema).names
        folders.append(f"| `{source}` | {CONTENTS[source]} | {', '.join(f'`{c}`' for c in columns)} | "
                       f"[{repo}](https://huggingface.co/datasets/{repo}) | {splits['train']:,} | "
                       f"{splits['val']:,} |")
    total = sum(n for splits in rows.values() for n in splits.values())
    return "\n".join([
        "---", "pretty_name: sgc pretraining corpus", "viewer: false", "---", "",
        "# sgc pretraining corpus", "",
        f"Text for pretraining a Python and C++ code model from scratch: {total:,} documents from "
        f"{len(rows)} sources. Every text has LF line ends and no byte-order mark; every Python file of the "
        "repository, synthetic and function sources parses as Python 3, repaired from Python 2 where it did not; "
        "exact copies are removed across all sources. "
        "Train rows sharing a 10-gram of whitespace-separated words with a prompt or reference program of the scored "
        "benchmarks (HumanEval+, MBPP+, MultiPL-E's HumanEval and MBPP in C++, HumanEval-X C++, HumanEval infilling) "
        "are removed; `decontamination.json` lists each (shard, row before removal, benchmark tasks).", "",
        "## Layout", "",
        "One folder per source, each with `train/` and `val/` Parquet shards. Validation holds 1% of each source: "
        "whole repositories in the repository folders (`stack_edu_*`, `ultradata_stack_v3_l2_*`, "
        "`ultradata_code_l2_*`) and `codexglue_python`, whole problems in `codeforces_cots`, documents elsewhere. A "
        "repository folder's shard holds whole repositories, each cut into samples of files that import one another "
        "(`sample`); any other folder's train rows are in `id` order, so its leading shards are a uniform sample.", "",
        "## Schema", "",
        "| Column | Meaning |", "|---|---|", *(f"| `{c}` | {m} |" for c, m in COLUMNS.items()), "",
        "## Folders", "",
        "| Folder | Contents | Columns | Source | Train rows | Val rows |", "|---|---|---|---|---|---|", *folders, "",
        "## Tokenizer", "",
        "`tokenizer/`: the byte-level BPE vocabulary the sgc model was trained with, `tokenizer.tiktoken` (each "
        "token's bytes in base64 and its rank) and `tokenizer.pattern` (the pre-tokenizer's split pattern).", ""])


def push() -> None:
    """Write the card; preupload the shards, the decontamination record, the tokenizer's two files and the card in
    batches of PREUPLOAD bytes; then replace the dataset's files in one commit (files no longer present are deleted,
    the rest added)."""
    (CORPUS / "README.md").write_text(card())
    api = HfApi()
    additions = [CommitOperationAdd(path_in_repo=f.relative_to(CURATED).as_posix(), path_or_fileobj=f)
                 for f in sorted(CURATED.glob("*/*/*.parquet"))]
    additions += [CommitOperationAdd(path_in_repo=f.relative_to(CORPUS).as_posix(), path_or_fileobj=f)
                  for f in (TOKENIZER / "tokenizer.tiktoken", TOKENIZER / "tokenizer.pattern")]
    additions += [CommitOperationAdd(path_in_repo=RECORD.name, path_or_fileobj=RECORD),
                  CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=CORPUS / "README.md")]
    group, size = [], 0
    for addition in additions:
        group.append(addition)
        size += addition.upload_info.size
        if size >= PREUPLOAD or addition is additions[-1]:
            api.preupload_lfs_files(DATASET, additions=group, repo_type="dataset")
            group, size = [], 0
    added = {a.path_in_repo for a in additions} | {".gitattributes"}
    deletions = [CommitOperationDelete(f) for f in api.list_repo_files(DATASET, repo_type="dataset") if f not in added]
    api.create_commit(DATASET, repo_type="dataset", operations=deletions + additions,
                      commit_message="sgc corpus, curated and decontaminated")


def pull() -> None:
    """Download REVISION's curated shards and decontamination record into CURATED and its tokenizer into TOKENIZER."""
    snapshot_download(DATASET, repo_type="dataset", revision=REVISION, local_dir=CURATED,
                      allow_patterns=[f"{source}/**" for source in SOURCES] + [RECORD.name])
    pull_tokenizer()


def pull_tokenizer() -> None:
    """Download REVISION's tokenizer alone into TOKENIZER. Evaluation needs it (eval/official_loop_stop.py decodes
    with it) but not the corpus."""
    snapshot_download(DATASET, repo_type="dataset", revision=REVISION, local_dir=CORPUS, allow_patterns=["tokenizer/*"])


if __name__ == "__main__":
    steps = {"push": push, "pull": pull, "pull_tokenizer": pull_tokenizer}
    parser = argparse.ArgumentParser(prog="python -m corpus.pretrain.publish",
                                     description="The curated corpus to the Hub and back.")
    parser.add_argument("step", choices=steps)
    steps[parser.parse_args().step]()
