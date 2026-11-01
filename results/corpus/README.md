---
pretty_name: nanochat pretraining corpus
viewer: false
---

# nanochat pretraining corpus

Text for pretraining a Python and C++ code model from scratch: 178,396,088 documents from 18 sources. Every text has LF line ends and no byte-order mark; every Python file parses as Python 3, repaired from Python 2 where it did not; exact copies are removed across all sources. Train rows sharing a 10-gram of whitespace-separated words with a prompt or reference program of the scored benchmarks (HumanEval+, MBPP+, MultiPL-E's HumanEval and MBPP in C++, HumanEval-X C++, HumanEval infilling) are removed; `decontamination.json` lists each (shard, row before removal, benchmark tasks).

## Layout

One folder per source, each with `train/` and `val/` Parquet shards. Validation holds 1% of each source: whole repositories in the repository folders (`stack_edu_*`, `ultradata_stack_v3_l2_*`, `ultradata_code_l2_*`) and `codexglue_python`, whole problems in `codeforces_cots`, documents elsewhere. A repository folder's shard holds whole repositories, each cut into samples of files that import one another (`sample`); any other folder's train rows are in `id` order, so its leading shards are a uniform sample.

## Schema

| Column | Meaning |
|---|---|
| `content` | the document's text |
| `repo_name` | its GitHub repository, `owner/name` |
| `relative_path` | the file's path in its repository |
| `sample` | within a shard, files sharing a number import one another; they are consecutive, in dependency order |
| `id` | the SHA-1 of `content` |
| `language` | the code's language, `Python` or `C++` |
| `problem_id` | the Codeforces problem |

## Folders

| Folder | Contents | Columns | Source | Train rows | Val rows |
|---|---|---|---|---|---|
| `stack_edu_python` | GitHub Python files filtered for educational value | `repo_name`, `relative_path`, `content`, `sample` | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 24,469,760 | 242,463 |
| `stack_edu_cpp` | GitHub C++ files filtered for educational value | `repo_name`, `relative_path`, `content`, `sample` | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 16,072,575 | 157,572 |
| `ultradata_stack_v3_l2_python` | GitHub Python files, quality-filtered | `repo_name`, `relative_path`, `content`, `sample` | [openbmb/UltraData-Stack-v3-L2](https://huggingface.co/datasets/openbmb/UltraData-Stack-v3-L2) | 17,440,055 | 183,818 |
| `ultradata_stack_v3_l2_cpp` | GitHub C++ files, quality-filtered | `repo_name`, `relative_path`, `content`, `sample` | [openbmb/UltraData-Stack-v3-L2](https://huggingface.co/datasets/openbmb/UltraData-Stack-v3-L2) | 12,365,009 | 122,010 |
| `ultradata_code_l2_python` | GitHub Python files, quality-filtered | `repo_name`, `relative_path`, `content`, `sample` | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | 41,954,868 | 425,751 |
| `ultradata_code_l2_cpp` | GitHub C++ files, quality-filtered | `repo_name`, `relative_path`, `content`, `sample` | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | 13,142,301 | 121,601 |
| `ultradata_code_l3_python` | synthetic Python exercises: a task, then its solution | `id`, `content`, `language` | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | 3,833,629 | 39,820 |
| `ultradata_code_l3_cpp` | synthetic C++ exercises: a task, then its solution | `id`, `content`, `language` | [openbmb/UltraData-Code](https://huggingface.co/datasets/openbmb/UltraData-Code) | 4,279,369 | 43,356 |
| `nemotron_rewriting_python` | GitHub Python files rewritten by an LLM (Qwen3-32B) | `id`, `content`, `language` | [nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | 18,020,706 | 182,625 |
| `nemotron_transpilation_cpp` | GitHub Python files transpiled into C++ by an LLM (Qwen3-32B) | `id`, `content`, `language` | [nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) | 5,125,846 | 51,878 |
| `codexglue_python` | Python functions with their docstrings (CodeSearchNet) | `id`, `repo_name`, `content`, `language` | [google/code_x_glue_ct_code_to_text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text) | 249,418 | 1,795 |
| `ultradata_math_l3_qa` | synthetic mathematics: questions and answers | `id`, `content` | [openbmb/UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | 1,073,272 | 10,842 |
| `ultradata_math_l3_conversation` | synthetic mathematics: conversations | `id`, `content` | [openbmb/UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | 667,967 | 6,714 |
| `ultradata_math_l3_multi_style` | synthetic mathematics: varied styles | `id`, `content` | [openbmb/UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | 678,935 | 6,839 |
| `ultradata_math_l3_textbook_exercise` | synthetic mathematics: textbook exercises | `id`, `content` | [openbmb/UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math) | 1,029,231 | 10,506 |
| `codeforces_cots` | Codeforces problems with finished DeepSeek-R1 reasoning traces and Python or C++ solutions | `id`, `problem_id`, `content` | [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) | 22,450 | 225 |
| `kaggle_notebooks` | Kaggle notebooks in Markdown: title, dataset description, the notebook as a Python script | `id`, `content`, `language` | [HuggingFaceTB/issues-kaggle-notebooks](https://huggingface.co/datasets/HuggingFaceTB/issues-kaggle-notebooks) | 573,674 | 5,815 |
| `climbmix` | English web text | `id`, `content` | [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | 15,625,499 | 157,894 |

## Tokenizer

`tokenizer/`: the byte-level BPE vocabulary the nanochat model was trained with, `tokenizer.tiktoken` (each token's bytes in base64 and its rank) and `tokenizer.pattern` (the pre-tokenizer's split pattern).
