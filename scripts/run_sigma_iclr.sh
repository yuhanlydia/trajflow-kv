#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
# No background jobs; one suite holds the process lock.
python -m tango_iclr.suite --config "${SIGMA_CONFIG:-configs/sigma_real_gui.yaml}" "$@"
