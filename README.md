# single-gpu-code-llm

sgc (Single-GPU Coder) is a 522M-parameter decoder-only transformer for Python and C++ code, pretrained from scratch
on one NVIDIA GH200 GPU, then post-trained with supervised fine-tuning (SFT) and reinforcement learning (RL). This
repository sets up inference and evaluation of the published models, and reproduces their training, for the paper
*One GPU, 0.1 Trillion Tokens: Pretraining a Sub-Billion-Parameter Code Model*.

## Models and data

| Hugging Face | Contents |
|---|---|
| [`iaroy/sgc-522m`](https://huggingface.co/iaroy/sgc-522m) | the final model, after RL (the default for inference) |
| [`iaroy/sgc-522m-sft`](https://huggingface.co/iaroy/sgc-522m-sft) | the model after SFT |
| [`iaroy/sgc-522m-base`](https://huggingface.co/iaroy/sgc-522m-base) | the pretrained model: code completion and fill-in-the-middle |
| [`iaroy/sgc-data`](https://huggingface.co/datasets/iaroy/sgc-data), revision `dbe57a81ba0314a501381da2331e598bc993be4c` | the curated pretraining corpus, its decontamination record and the tokenizer |

SFT and RL use four further datasets: [`nvidia/Nemotron-Cascade-SFT-Stage-1`](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-1),
[`nvidia/Nemotron-Cascade-SFT-Stage-2`](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2),
[`nvidia/Nemotron-SFT-Competitive-Programming-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2)
and [`KodCode/KodCode-V1-SFT-R1`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1).

## Requirements

Every run used one NVIDIA GH200 (96 GB, aarch64; Lambda `gpu_1x_gh200`) with Docker, the NVIDIA driver and the
NVIDIA Container Toolkit, and sudo for setup. All software runs in three Docker images, with base images pinned by
digest and every dependency by version:

| Image | Contents | Defined in |
|---|---|---|
| `sgc` | training: NGC PyTorch 26.09 (PyTorch 2.14.0a0), uv 0.13.6, Rust 1.99.1, Python 3.12 packages | `docker/Dockerfile`, `pyproject.toml`, `uv.lock` |
| `sgc-eval` | inference and evaluation: vLLM 0.32.0, EvalPlus, MultiPL-E, lm-evaluation-harness, human-eval-infilling (pinned commits) | `docker/Dockerfile.eval` |
| `sgc-rl` | RL: `sgc-eval` with matplotlib | `docker/Dockerfile.rl` |


## Setup

```bash
git clone https://github.com/indrajeetadityaroy9/single-gpu-code-llm && cd single-gpu-code-llm
scripts/setup.sh
```

`scripts/setup.sh` writes the CDI spec that gives containers the GPU, locks the SM clock (run it again after a
reboot), adds you to the docker group, builds the three images and downloads the tokenizer from `iaroy/sgc-data`.
No Hugging Face token is needed. Log in again afterwards for the docker group. Every other script assumes the images
exist, and everything a run writes goes to `artifacts/` (not tracked by git).

## Inference

```bash
scripts/serve.sh                      # iaroy/sgc-522m
scripts/serve.sh iaroy/sgc-522m-sft   # any Hugging Face id or local model directory
```

`scripts/serve.sh` downloads the model to `artifacts/models/<owner>/<name>` and starts vLLM's OpenAI-compatible
server in `sgc-eval` at `http://127.0.0.1:8000/v1`, in the container `sgc-serve` (`docker stop sgc-serve` stops it).
Its `sgc` reasoning parser returns each response's reasoning (`reasoning`) apart from its answer (`content`). Further
arguments go to `vllm serve`, such as `--gpu-memory-utilization 0.3` beside another GPU job. It prints a request to try:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "sgc",
  "messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
  "max_tokens": 8192, "temperature": 0.6, "top_p": 0.95
}' | python3 -c 'import json, sys; m = json.load(sys.stdin)["choices"][0]["message"]
print("reasoning:", m["reasoning"], "", "answer:", m["content"], sep="\n")'
```

The post-trained models take chat messages; the pretrained model completes code (`/v1/completions`).

## Evaluating the published models

```bash
scripts/eval_pretrained.sh iaroy/sgc-522m-base    # EvalPlus, MultiPL-E C++, lm-eval MBPP 3-shot, HumanEval infilling
scripts/eval_posttrained.sh iaroy/sgc-522m-sft    # EvalPlus with the chat prompt, greedy and 20 samples
scripts/eval_posttrained.sh iaroy/sgc-522m
scripts/eval_kodcode.sh iaroy/sgc-522m            # pass@1 on the 512 held-out KodCode problems
scripts/termination_probe.sh iaroy/sgc-522m-sft   # how often the responses end their reasoning
```

Each writes `artifacts/evals/<model path under artifacts/>`, here `artifacts/evals/models/iaroy/<name>/`, with the
benchmarks' official outputs and a summary. The records of the paper's own evaluations are in `results/runs/`; their
EvalPlus files are named `nanochat_openai_*`, after the model's development name, where a new run writes
`sgc_openai_*`. The KodCode validation and the termination probe take their prompts from the SFT corpus's held-out
split and the RL prompt pools, so they need those built first (Data, below).

## Reproducing training

### Data

Download the curated pretraining corpus (166 GB), then tokenize and pack it:

```bash
scripts/container.sh python -m corpus.pretrain.publish pull
scripts/container.sh python -m corpus.tokenize   # token shards for both stages (8,192 and 32,768-token context)
scripts/container.sh python -m corpus.pack       # packed training rows with fill-in-the-middle examples
```

On the GH200 the GPU's memory is a NUMA node of the system, and these CPU-heavy steps can leave Linux page cache in it.
Before training, free it with `sync && echo 1 | sudo tee /proc/sys/vm/drop_caches`.

To rebuild `iaroy/sgc-data` itself from its sources, run `corpus.pretrain.prepare`, `corpus.pretrain.curate` and
`corpus.decontaminate`, in that order, with `scripts/container.sh python -m` and `HF_TOKEN` set (one source,
`nvidia/Nemotron-Pretraining-Code-v2`, is gated). Tokenization uses the shipped tokenizer in `iaroy/sgc-data`, which
`corpus/train_tokenizer.py` trained; retraining it needs an empty `artifacts/corpus/tokenizer` and does not reproduce
the shipped tokenizer byte for byte.

Download the SFT sources, then build the SFT corpus and the RL prompt pools:

```bash
H="scripts/container.sh hf download --type dataset"
$H nvidia/Nemotron-Cascade-SFT-Stage-1 --include "code/*.jsonl" --local-dir artifacts/posttrain/nvidia/Nemotron-Cascade-SFT-Stage-1
$H nvidia/Nemotron-Cascade-SFT-Stage-2 --include "code/*.jsonl" --local-dir artifacts/posttrain/nvidia/Nemotron-Cascade-SFT-Stage-2
$H nvidia/Nemotron-SFT-Competitive-Programming-v2 --include "data/competitive_programming_*.jsonl" --include "data/exercism.jsonl" \
  --local-dir artifacts/posttrain/nvidia/Nemotron-SFT-Competitive-Programming-v2
$H KodCode/KodCode-V1-SFT-R1 --include "data/train-*.parquet" --local-dir artifacts/posttrain/KodCode/KodCode-V1-SFT-R1
scripts/container.sh python -m corpus.posttrain.build   # the SFT corpus
scripts/container.sh python -m rl.prompts               # the RL prompt pools, from KodCode
```

### Runs

Run in this order from the repository root. `python run.py <experiment>` runs one experiment of `run.py` and writes
`artifacts/runs/<run>/`; it fails if that directory exists. The calibrations write their results to `results/`
(`results/pretrain/peak_lr_d24.jsonl`, `results/posttrain/sft_lr.jsonl`), from which pretraining and SFT take their
learning rates.

| Step | Command | Output | Time on one GH200 |
|---|---|---|---|
| 1. Peak learning-rate calibration (3 short runs) | `scripts/container.sh python run.py peak_lr` | `artifacts/runs/peak_lr-{1.0,2.0,4.0}` | |
| 2. Pretraining (69,874 steps, 103B tokens) | `tmux new -d -s run 'scripts/container.sh python run.py pretrain'` | `artifacts/runs/pretrain` | 76.3 h |
| 3. Export and evaluation of the pretrained model | see below | `artifacts/evals/models/pretrain` | |
| 4. SFT learning-rate calibration (3 short runs) | `scripts/container.sh python run.py sft_lr` | `artifacts/runs/sft_lr-*` | |
| 5. SFT (7,676 steps) | `scripts/container.sh python run.py sft` | `artifacts/runs/sft` | 11.2 h |
| 6. Termination probes, RL and evaluation of the SFT and RL models | `scripts/post_sft.sh` | `artifacts/models/sft`, `artifacts/runs/rl`, `artifacts/evals/` | RL: 2 h |

The evaluation scripts take a model in the Hugging Face format; step 3 exports the final pretraining checkpoint
(`--chat` would add the chat template of the post-trained models):

```bash
scripts/container.sh python -m inference.export_hf artifacts/runs/pretrain/checkpoint-69874.pt artifacts/models/pretrain
scripts/eval_pretrained.sh artifacts/models/pretrain
```

`scripts/post_sft.sh` exports the SFT run's final checkpoint (step 7,676) with the chat template to
`artifacts/models/sft`, measures the loops' temperature sensitivity at step 2,000, runs the termination probe and the
KodCode validation of the SFT model, RL for two hours (`artifacts/runs/rl`, its last policy in `final/`), and then the
KodCode validation of RL's last policy and EvalPlus with the chat prompt of both models.

The paper's training figures come from the pretraining run's records:

```bash
scripts/container.sh python scripts/plot_training.py artifacts/runs/pretrain   # writes figures/
```
