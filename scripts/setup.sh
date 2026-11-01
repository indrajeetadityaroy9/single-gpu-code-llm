#!/bin/bash
# Prepare a GH200 host that has Docker, the NVIDIA driver and the NVIDIA Container Toolkit: write the CDI spec (GPU
# access for containers), lock the SM clock, add the user to the docker group, build the three images (sgc, sgc-eval,
# sgc-rl) and download the corpus tokenizer from iaroy/sgc-data, which the evaluation drivers read. Every other script
# assumes the images exist. A reboot resets the clock lock: run the script again after one.
#
#   scripts/setup.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# Lock the SM clock at its maximum: under a power cap the driver otherwise lowers it. This affects speed only, not
# numerics.
MAX_SM_CLOCK=$(nvidia-smi --id=0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits)
sudo nvidia-smi -lgc "$MAX_SM_CLOCK,$MAX_SM_CLOCK"
sudo usermod -aG docker "$USER"
# Group membership applies at the next login. Until then, run docker through `sg docker`; the exported function also
# reaches the script called below.
if ! docker info > /dev/null 2>&1; then
  docker() { sg docker -c "$(printf '%q ' docker "$@")"; }
  export -f docker
fi
# artifacts/ (or the disk it links to) must exist before a container mounts it
mkdir -p "$ROOT/artifacts"
# The training image's context is the repository, filtered to pyproject.toml and uv.lock by
# docker/Dockerfile.dockerignore; sgc-rl is built FROM sgc-eval.
docker build -f "$ROOT/docker/Dockerfile" -t "$IMAGE_TRAIN" "$ROOT"
docker build -f "$ROOT/docker/Dockerfile.eval" -t "$IMAGE_EVAL" "$ROOT/docker"
docker build -f "$ROOT/docker/Dockerfile.rl" -t "$IMAGE_RL" "$ROOT/docker"
"$ROOT/scripts/container.sh" python -m corpus.pretrain.publish pull_tokenizer
echo "Log in again for the docker group to apply."
