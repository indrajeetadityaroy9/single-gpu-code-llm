#!/bin/bash
# Run a command in the training image (sgc) on the GPU, as the invoking user, with this checkout mounted at
# /workspace/sgc (paths.py ROOT).
#
#   scripts/container.sh <command...>     for example: scripts/container.sh python run.py pretrain
#
# HF_TOKEN is passed through when set. The published dataset is public, but rebuilding the corpus reads gated sources
# (corpus/pretrain/sources.py), and publishing writes to the Hub.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
"${GPU[@]}" -e HF_TOKEN "$IMAGE_TRAIN" "$@"
