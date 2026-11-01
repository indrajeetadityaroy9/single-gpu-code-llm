#!/bin/bash
# Evaluate a pretrained model with each benchmark's official tool, unmodified (paper §6.1, §6.2): EvalPlus (HumanEval+,
# MBPP+), MultiPL-E C++ (greedy and sampled), lm-evaluation-harness MBPP 3-shot, and HumanEval infilling. Each tool
# generates with its own vLLM 0.32.0 engine in the sgc-eval image (docker/Dockerfile.eval), with its own share of GPU
# memory, and its judge runs in a CPU container. eval/official.py then writes the summary.
#
#   scripts/eval_pretrained.sh <Hugging Face id or model directory>    for example iaroy/sgc-522m-base
#
# A checkpoint is first exported to the Hugging Face format (README.md). Outputs in artifacts/evals/<model path under
# artifacts/> (scripts/lib.sh eval_dir): evalplus/, multipl-e/{greedy,sampled}/, lm-eval/, infilling/, tools.json
# (tool and library versions), summary.json.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODEL=$(model_dir "$1")
OUT=$(eval_dir "$MODEL")
C=$(in_container "$OUT") M=/model  # the generators read the model at /model
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"  # a rerun fails here instead of mixing two runs' samples
mkdir -p "$OUT"/{lm-eval,infilling} "$OUT"/multipl-e/{greedy,sampled}/{humaneval,mbpp}-cpp
GEN=("${VLLM[@]}" "${LIMITS[@]}" -v "$MODEL:$M:ro")
JUDGE=("${CPU[@]}" "$IMAGE_EVAL") JUDGE_WORKERS=32  # a judge step's test processes
"${RUN[@]}" "$IMAGE_EVAL" cp /opt/tools.json "$C/tools.json"

# EvalPlus: greedy, 768 new tokens, its own stop strings and sanitizer (codegen.py, provider/vllm.py)
for dataset in humaneval mbpp; do
  "${GEN[@]}" "$IMAGE_EVAL" python -m evalplus.codegen --model $M --dataset $dataset --backend vllm --greedy \
    --trust_remote_code True --root "$C/evalplus"
  "${JUDGE[@]}" bash -c "python -m evalplus.evaluate --dataset $dataset \
    --samples \$(ls $C/evalplus/$dataset/*.jsonl | grep -v '\.raw\.jsonl$')"
done

# MultiPL-E C++, 1,024 new tokens: greedy (temperature 0, one completion) and sampled (temperature 0.2, top-p 0.95,
# 20 completions). The judge runs one problem file per main.py process, since main.py --dir judges problems serially.
for dataset in humaneval mbpp; do
  for mode in "greedy 0 1" "sampled 0.2 20"; do
    read -r name temperature completions <<< "$mode"
    workers=$((completions > 4 ? 4 : completions))
    "${GEN[@]}" -w /opt/MultiPL-E "$IMAGE_EVAL" python automodel_vllm.py --name $M \
      --root-dataset $dataset --lang cpp --temperature "$temperature" --top-p 0.95 \
      --completion-limit "$completions" --output-dir "$C/multipl-e/$name/$dataset-cpp"
    "${JUDGE[@]}" bash -c "ls $C/multipl-e/$name/$dataset-cpp/*.json.gz | grep -v '\.results\.json\.gz$' | \
      xargs -P $((JUDGE_WORKERS / workers)) -I{} python \$MULTIPL_E/evaluation/src/main.py --file {} \
      --output-dir $C/multipl-e/$name/$dataset-cpp --max-workers $workers > /dev/null"
  done
done
"${JUDGE[@]}" bash -c "python /opt/MultiPL-E/pass_k.py $C/multipl-e/*/*" | tee "$OUT/multipl-e/pass_k.csv"

# lm-evaluation-harness MBPP 3-shot (greedy, stop at "[DONE]", 256 new tokens; the tokenizer adds BOS) as the task
# mbpp_lf (eval/official_lm_eval/): mbpp with the prompt's line endings normalized to LF, as in the training corpus
"${GEN[@]}" -e HF_ALLOW_CODE_EVAL=1 "$IMAGE_EVAL" lm_eval --model vllm \
  --model_args pretrained=$M,trust_remote_code=True,add_bos_token=True --tasks mbpp_lf \
  --include_path $W/eval/official_lm_eval --confirm_run_unsafe_code --log_samples --output_path "$C/lm-eval"

# HumanEval infilling: one greedy middle per task in PSM order (eval/official_infilling.py), judged by the package's
# evaluate_functional_correctness (3 s per sample; of its default k "1,10,100", only pass@1 applies to one sample)
"${GEN[@]}" "$IMAGE_EVAL" python -m eval.official_infilling $M "$C/infilling"
for samples in "$OUT"/infilling/*-psm.jsonl; do
  "${JUDGE[@]}" python -m human_eval_infilling.evaluate_functional_correctness \
    --benchmark_name "$(basename "$samples" -psm.jsonl)" --sample_file "$(in_container "$samples")" \
    --n_workers "$JUDGE_WORKERS"
done

"${CPU[@]}" "$IMAGE_TRAIN" python -m eval.official "$C"
