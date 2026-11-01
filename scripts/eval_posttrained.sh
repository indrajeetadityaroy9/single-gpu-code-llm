#!/bin/bash
# Evaluate a post-trained (SFT or RL) model on HumanEval+ and MBPP+ with EvalPlus's chat prompt (paper §6.1, §6.3,
# §6.4). The model is served by scripts/serve.sh, whose "sgc" reasoning parser splits each response at its last
# "</think>" into the reasoning and the answer; EvalPlus scores the code in the answer.
#
#   scripts/eval_posttrained.sh <Hugging Face id or model directory>    for example iaroy/sgc-522m
#
# EvalPlus's OpenAI client (eval/official_evalplus.py, which also records how each response ended: its finish reason
# and whether "</think>" closed the reasoning) generates greedy (one sample, temperature 0) and sampled (20 samples at
# eval/official_termination.py SAMPLING's temperature; top-p 0.95, as EvalPlus's openai_request.make_request sets it),
# with max_tokens the export's context length (32,768) in every request (vLLM caps it at the context length minus the
# prompt). EvalPlus's evaluator judges them, and
# eval/official_reasoning.py summarizes: pass@1, the share of responses that closed their reasoning, pass@1 given
# closure, and the loop stops. The server stops reasoning loops (eval/official_loop_stop.py).
# The server takes 0.45 of the GPU's memory, as in the reported runs: the share sizes the KV cache, which can change
# vLLM's batching and hence even greedy outputs. The KV cache takes 2 x 24 layers x 6 heads x 128 x 2 bytes = 72 KiB
# per token, so this share holds 20 sequences (EvalPlus's number of samples) of about 28K tokens beside the weights.
#
# Outputs in artifacts/evals/<model path under artifacts/> (scripts/lib.sh eval_dir): evalplus/ (each samples file
# with its .endings.jsonl), summary.json.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODEL=$(model_dir "$1")
OUT=$(eval_dir "$MODEL")
C=$(in_container "$OUT") URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"  # shared with scripts/termination_probe.sh and scripts/eval_kodcode.sh
mkdir "$OUT/evalplus"  # a rerun fails here instead of resuming EvalPlus's samples beside appended endings
# The sampled mode's temperature and the export's context length (EvalPlus's max_new_tokens); the image's entrypoint
# prints the NGC banner first, so the values are the last line
read -r TEMPERATURE CONTEXT <<< "$("${CPU[@]}" -v "$MODEL":/model:ro "$IMAGE_TRAIN" python -c '
import json; from eval.official_termination import SAMPLING
print(SAMPLING["temperature"], json.load(open("/model/config.json"))["max_position_embeddings"])' | tail -n 1)"

"$ROOT/scripts/serve.sh" "$MODEL" --gpu-memory-utilization 0.45 \
  --logits-processors eval.official_loop_stop:LoopStopLogitsProcessor > /dev/null
trap 'docker rm -f "$SERVER" > /dev/null' EXIT  # removed at once, so that the next server can take the name
CLIENT=("${CPU[@]}" --network host -e OPENAI_BASE_URL="$URL/v1" -e OPENAI_API_KEY=none -e OPENAI_KEY=none "$IMAGE_EVAL")
EVALPLUS=(--model sgc --backend openai --base_url "$URL/v1" --max_new_tokens "$CONTEXT" --root "$C/evalplus")
for dataset in humaneval mbpp; do
  # the samples files: sgc_openai_temp_0.0.jsonl (greedy), sgc_openai_temp_$TEMPERATURE.jsonl (sampled)
  "${CLIENT[@]}" python -m eval.official_evalplus --dataset $dataset --greedy "${EVALPLUS[@]}"
  "${CLIENT[@]}" python -m eval.official_evalplus --dataset $dataset --n_samples 20 --temperature "$TEMPERATURE" \
    "${EVALPLUS[@]}"
  for temperature in 0.0 "$TEMPERATURE"; do
    "${CPU[@]}" "$IMAGE_EVAL" python -m evalplus.evaluate --dataset $dataset \
      --samples "$C/evalplus/$dataset/sgc_openai_temp_$temperature.jsonl"
  done
done
"${CPU[@]}" "$IMAGE_TRAIN" python -m eval.official_reasoning "$C"
