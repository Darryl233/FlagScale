#!/usr/bin/env bash
# Qwen3.5-35B-A3B 64K smoke/profile launch on one 16-NPU node.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_dir}"

export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"

flagscale run \
  --config-path examples/qwen35/conf \
  --config-name train_64k \
  "$@"
