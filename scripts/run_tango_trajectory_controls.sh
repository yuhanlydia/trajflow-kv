#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CONFIG DATA_PATH WARM_START_CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

config=$1
data_path=$2
checkpoint=$3
output_root=$4
python_bin=${PYTHON_BIN:-.venv/bin/python}

common=(
  --config "$config"
  --data-path "$data_path"
  --projector-checkpoint "$checkpoint"
)

# Return-label controls: observed, within-task shuffle, global seeded random,
# and sign-flipped consequence. Randomness is derived from config seed + epoch.
for mode in observed shuffle random sign_flip; do
  "$python_bin" -m trajflow_kv.train "${common[@]}" \
    --return-mode "$mode" --output-dir "$output_root/return_$mode"
done

# Trajectory-vs-token controls. H means the first H executed transitions.
for horizon in 1 3 5 full; do
  "$python_bin" -m trajflow_kv.train "${common[@]}" \
    --return-mode observed --trajectory-horizon "$horizon" \
    --output-dir "$output_root/horizon_$horizon"
done

for selection in first final; do
  "$python_bin" -m trajflow_kv.train "${common[@]}" \
    --return-mode observed --step-selection "$selection" \
    --output-dir "$output_root/only_$selection"
done

"$python_bin" -m trajflow_kv.train "${common[@]}" \
  --return-mode observed --remove-history \
  --output-dir "$output_root/remove_history"
