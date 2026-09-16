#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use the environment's Python; one GPU process at a time.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
python -m tango_iclr.suite --config "${TANGO_CONFIG:-configs/tango_iclr_release.yaml}" "$@"
