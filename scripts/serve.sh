#!/bin/bash
# Serve an sgc model with vLLM's OpenAI-compatible server (vLLM 0.32.0, in the sgc-eval image) at
# http://127.0.0.1:$PORT/v1 (scripts/lib.sh).
#
#   scripts/serve.sh [<model: Hugging Face id or local directory, default iaroy/sgc-522m>] [<vllm serve flags>...]
#
# The models are iaroy/sgc-522m (the RL model), iaroy/sgc-522m-sft and iaroy/sgc-522m-base (the pretrained model,
# which has no chat format: use /v1/completions). A local directory must be in the format inference/export_hf.py
# writes; a Hub model is downloaded to artifacts/models/<owner>/<name>. The server loads the model's own code
# (modeling_sgc.py, --trust-remote-code), runs in bf16 and names the model "sgc". Its "sgc" reasoning parser
# (eval/official_reasoning_parser.py) splits each chat response at the last "</think>": the reasoning is returned in
# the message's `reasoning` field and the answer in `content`. scripts/eval_posttrained.sh adds its own flags.
# The server runs detached in the container $SERVER (scripts/lib.sh); stop it with `docker stop $SERVER`.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODEL=$(model_dir "${1:-iaroy/sgc-522m}")
"${VLLM[@]}" -d --name "$SERVER" --network host -v "$MODEL":/model:ro "$IMAGE_EVAL" vllm serve /model --port "$PORT" \
  --served-model-name sgc --trust-remote-code --dtype bfloat16 --reasoning-parser sgc "${@:2}" > /dev/null
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  docker inspect "$SERVER" > /dev/null 2>&1 || { echo "vllm serve exited" >&2; exit 1; }
  sleep 5
done
cat << EOF
Serving $MODEL as "sgc" at http://127.0.0.1:$PORT/v1. Example request:

curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "sgc",
  "messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
  "max_tokens": 8192, "temperature": 0.6, "top_p": 0.95
}' | python3 -c 'import json, sys; m = json.load(sys.stdin)["choices"][0]["message"]
print("reasoning:", m["reasoning"], "", "answer:", m["content"], sep="\n")'
EOF
