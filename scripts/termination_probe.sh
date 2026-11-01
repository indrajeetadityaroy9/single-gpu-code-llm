#!/bin/bash
# Natural-termination probe of a post-trained model (eval/official_termination.py; paper §6.1, §6.4): sample a fixed
# set of HumanEval+ prompts and held-out SFT validation prompts, and count how the responses end (reasoning closed with
# "</think>" or not; ended, at the length limit, or stopped as a loop). The responses are sampled by vLLM's offline
# engine on the GPU, then scored in a separate CPU container, never in the vLLM process.
#
#   scripts/termination_probe.sh <Hugging Face id or model directory>
#
# Outputs in artifacts/evals/<model path under artifacts/> (scripts/lib.sh eval_dir): termination.generated.json (the
# responses), termination.json (scored).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODEL=$(model_dir "$1")
OUT=$(eval_dir "$MODEL")
C=$(in_container "$OUT")
mkdir -p "$OUT"
"${VLLM[@]}" -v "$MODEL":/model:ro "$IMAGE_EVAL" python -m eval.official_termination /model \
  "$C/termination.generated.json"
"${CPU[@]}" "$IMAGE_EVAL" python -m eval.official_termination --score "$C/termination.generated.json" \
  "$C/termination.json"
