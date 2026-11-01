# shellcheck shell=bash disable=SC2034  # the arrays and names below are used by the scripts that source this file
# Shared by the scripts in this directory: the repository root, the Docker image names (built by scripts/setup.sh),
# the `docker run` flag sets for CPU, GPU and vLLM containers, and model downloads from the Hugging Face Hub.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
W=/workspace/sgc  # the checkout's path inside every container (paths.py ROOT)
IMAGE_TRAIN=sgc IMAGE_EVAL=sgc-eval IMAGE_RL=sgc-rl
PORT=8000 SERVER=sgc-serve  # the vLLM server's port on the host network and its container (scripts/serve.sh)

# Every container: removed on exit; host IPC namespace (PyTorch and vLLM share tensors through /dev/shm); an init
# process that reaps child processes; no core dumps; the invoking user's identity, so outputs are owned by them; the
# checkout at $W and the caches under artifacts/cache. artifacts/ may be a symbolic link to another disk; it is mounted
# at its real path.
RUN=(docker run --rm --ipc=host --init --ulimit core=0 --user "$(id -u):$(id -g)" -e HOME=/tmp
     -e XDG_CACHE_HOME="$W/artifacts/cache" -e HF_HOME="$W/artifacts/cache/huggingface" -e PYTHONPATH="$W"
     -v "$ROOT:$W" -v "$(realpath "$ROOT/artifacts"):$W/artifacts" -w "$W")
# NVIDIA's documented run flags for PyTorch containers: unlimited locked memory, and a 64 MiB stack with an unlimited
# hard limit. EvalPlus's reliability_guard raises RLIMIT_STACK, which needs the unlimited hard limit, and the corpus
# crate sizes its worker threads' stacks from the soft limit (corpus/crate/lib.rs).
LIMITS=(--ulimit memlock=-1 --ulimit stack=67108864:-1)
# CPU containers: memory on NUMA node 0, the CPU's (node 1 is the GPU's memory), and one BLAS and OpenMP thread per
# process. evalplus.evaluate's reliability_guard caps each test process's address space (RLIMIT_AS), which counts the
# memory mappings the process inherits; per-core thread pools on 64 cores inflate them until the tests with the
# largest allocations fail with MemoryError (HumanEval/15's plus input [1000000]).
CPU_FLAGS=(--cpuset-mems=0 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1)
CPU=("${RUN[@]}" "${LIMITS[@]}" "${CPU_FLAGS[@]}")
GPU=("${RUN[@]}" "${LIMITS[@]}" --device nvidia.com/gpu=all)
# A vLLM engine on the GPU. eval/official_site on PYTHONPATH loads its sitecustomize.py into every Python process, which
# registers the "sgc" reasoning parser.
VLLM=("${RUN[@]}" --device nvidia.com/gpu=all -e VLLM_CACHE_ROOT="$W/artifacts/cache/vllm"
      -e PYTHONPATH="$W/eval/official_site:$W")

in_container() {  # <host path under ROOT>: the same path inside a container
  echo "${1/#$ROOT/$W}"
}

model_dir() {  # <Hugging Face id or local directory>: the model's local directory. A Hub id is downloaded into
  # artifacts/models/<owner>/<name> by the hf command line in the evaluation image.
  if [ -d "$1" ]; then
    (cd "$1" && pwd)
  else
    mkdir -p "$ROOT/artifacts/models/$1"
    "${RUN[@]}" "$IMAGE_EVAL" hf download "$1" --local-dir "$W/artifacts/models/$1" >&2
    echo "$ROOT/artifacts/models/$1"
  fi
}

eval_dir() {  # <model directory under artifacts/>: its evaluation outputs' directory, artifacts/evals/<the same path>
  echo "$ROOT/artifacts/evals/${1#"$ROOT"/artifacts/}"
}
