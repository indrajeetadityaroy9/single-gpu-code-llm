#!/bin/bash
# KodCode validation pass@1 of a post-trained model (paper §5.2.2, §6.1): 4 responses at temperature 0.6 to each of the
# 512 held-out KodCode problems of the SFT validation split (paper §2.3), sampled and run against their tests by the RL
# sampler's engine (rl/sampler.py --validate), as RL's own evaluations do. The tests run the model's code, so the
# container has no network.
#
#   scripts/eval_kodcode.sh <Hugging Face id or model directory>
#
# Output: artifacts/evals/<model path under artifacts/>/kodcode.json (scripts/lib.sh eval_dir).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODEL=$(model_dir "$1")
OUT=$(eval_dir "$MODEL")
mkdir -p "$OUT"
"${VLLM[@]}" --network none -e HF_HUB_OFFLINE=1 -v "$MODEL":/model:ro "$IMAGE_RL" python -m rl.sampler --validate \
  /model "$(in_container "$OUT")/kodcode.json"
