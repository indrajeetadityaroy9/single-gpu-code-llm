#!/bin/bash
# Everything after SFT that produces the reported post-training results (paper §5.2.1, §5.2.2, §6.3, §6.4), in order:
#   1. Export the SFT run's final checkpoint with the chat template (inference/export_hf.py --chat) to
#      artifacts/models/sft, and its step-2000 checkpoint to artifacts/models/sft-2000.
#   2. The loops' temperature sensitivity at step 2000 (eval/official_termination.py --sensitivity).
#   3. The SFT model's natural-termination probe (scripts/termination_probe.sh) and KodCode validation
#      (scripts/eval_kodcode.sh), the baseline RL's evaluations compare against.
#   4. RL from the final checkpoint: two containers share the GPU for the trainer's time budget (rl/trainer.py BUDGET),
#      the trainer (rl/trainer.py, training image, native model) and the sampler (rl/sampler.py, RL image: vLLM
#      engines that sample responses, and the KodCode tests that score them). They exchange policies and batches
#      through files in /dev/shm/sgc-rl (host IPC namespace), cleared first. Both run without a network, since the
#      tests run the model's code; if either fails, both are stopped. Output in artifacts/runs/rl: train.jsonl, and
#      the last policy in final/ (in the export's format).
#   5. RL's last policy's KodCode validation, and EvalPlus with the chat prompt (scripts/eval_posttrained.sh) of the
#      SFT model and of RL's last policy.
# Evaluation outputs in artifacts/evals/models/sft, models/sft-2000 and runs/rl/final.
#
#   scripts/post_sft.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
RUN_DIR=$ROOT/artifacts/runs/sft RL=$ROOT/artifacts/runs/rl X=/dev/shm/sgc-rl
FINAL=$(printf '%s\n' "$RUN_DIR"/checkpoint-*.pt | sort -V | tail -1)
SFT=$ROOT/artifacts/models/sft SFT_2000=$ROOT/artifacts/models/sft-2000
EXPORT=("${CPU[@]}" "$IMAGE_TRAIN" python -m inference.export_hf)
"${EXPORT[@]}" "$(in_container "$FINAL")" "$(in_container "$SFT")" --chat
"${EXPORT[@]}" "$(in_container "$RUN_DIR/checkpoint-2000.pt")" "$(in_container "$SFT_2000")" --chat

mkdir -p "$(eval_dir "$SFT_2000")"
"${VLLM[@]}" -v "$SFT_2000":/model:ro "$IMAGE_EVAL" python -m eval.official_termination --sensitivity /model \
  "$(in_container "$(eval_dir "$SFT_2000")")/sensitivity.json"
"$ROOT/scripts/termination_probe.sh" "$SFT"
"$ROOT/scripts/eval_kodcode.sh" "$SFT"

rm -rf "$X"
ISOLATED=(--network none -e HF_HUB_OFFLINE=1)
"${GPU[@]}" "${ISOLATED[@]}" --name sgc-rl-trainer "$IMAGE_TRAIN" python -m rl.trainer "$(in_container "$FINAL")" \
  "$(in_container "$SFT")" "$X" "$(in_container "$RL")" &
"${VLLM[@]}" "${ISOLATED[@]}" --name sgc-rl-sampler "$IMAGE_RL" python -m rl.sampler "$(in_container "$SFT")" "$X" &
wait -n || { docker rm -f sgc-rl-trainer sgc-rl-sampler > /dev/null; exit 1; }
wait

"$ROOT/scripts/eval_kodcode.sh" "$RL/final"
"$ROOT/scripts/eval_posttrained.sh" "$SFT"
"$ROOT/scripts/eval_posttrained.sh" "$RL/final"
