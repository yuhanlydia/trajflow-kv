#!/usr/bin/env bash
set -euo pipefail

export MINIWOB_URL="${MINIWOB_URL:-file:///root/miniwob-plusplus/miniwob/html/miniwob/}"
checkpoint="${1:-outputs/gonogo/return_fork_click_e3_coord_e2/kv_projectors.pt}"
output_dir="${2:-outputs/gonogo}"

run_arm() {
  local label="$1"
  shift
  .venv/bin/python scripts/evaluate_miniwob_online.py \
    --tasks click-button click-button-sequence click-tab \
    --seeds 401 402 403 --max-steps 3 "$@" \
    --output "$output_dir/miniwob_${label}_s401_403.json"
  .venv/bin/python scripts/evaluate_miniwob_online.py \
    --tasks click-checkboxes --seeds 404 405 406 407 408 --max-steps 8 "$@" \
    --output "$output_dir/miniwob_checkboxes_${label}_s404_408.json"
  .venv/bin/python scripts/evaluate_miniwob_online.py \
    --tasks click-option --seeds 409 410 411 412 413 --max-steps 8 "$@" \
    --output "$output_dir/miniwob_option_${label}_s409_413.json"
  .venv/bin/python scripts/evaluate_miniwob_online.py \
    --tasks click-color --seeds 414 415 416 417 418 --max-steps 1 "$@" \
    --output "$output_dir/miniwob_color_${label}_s414_418.json"
}

run_arm base
run_arm kv --checkpoint "$checkpoint"
.venv/bin/python scripts/summarize_miniwob_pair.py \
  --baseline \
    "$output_dir/miniwob_base_s401_403.json" \
    "$output_dir/miniwob_checkboxes_base_s404_408.json" \
    "$output_dir/miniwob_option_base_s409_413.json" \
    "$output_dir/miniwob_color_base_s414_418.json" \
  --candidate \
    "$output_dir/miniwob_kv_s401_403.json" \
    "$output_dir/miniwob_checkboxes_kv_s404_408.json" \
    "$output_dir/miniwob_option_kv_s409_413.json" \
    "$output_dir/miniwob_color_kv_s414_418.json" \
  --output "$output_dir/miniwob_paired_summary.json"
