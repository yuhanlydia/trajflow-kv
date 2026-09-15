#!/usr/bin/env bash
set -euo pipefail

export MINIWOB_URL="${MINIWOB_URL:-file:///root/miniwob-plusplus/miniwob/html/miniwob/}"
warm="${1:-outputs/gonogo/initial_v_l8/kv_projectors.pt}"
out="${2:-outputs/gonogo}"
train_data="data/miniwob/random_train_s501_510.jsonl"
heldout_data="data/miniwob/random_heldout_s511_515.jsonl"

.venv/bin/python scripts/collect_miniwob_random.py \
  --tasks click-button click-button-sequence click-tab click-color \
  --seeds 501 502 503 504 505 506 507 508 509 510 \
  --output "$train_data"
.venv/bin/python scripts/collect_miniwob_random.py \
  --tasks click-button click-button-sequence click-tab click-color \
  --seeds 511 512 513 514 515 --output "$heldout_data"

for objective in return shuffle ce; do
  extra=()
  case "$objective" in
    shuffle) extra+=(--return-mode shuffle) ;;
    ce) extra+=(--return-mode zero --positive-action-only --lambda-action 0.01) ;;
  esac
  .venv/bin/python -m trajflow_kv.train \
    --config configs/qwen_androidworld.yaml --data-path "$train_data" \
    --output-dir "$out/miniwob_${objective}_v_r8_l8_a8_e10" \
    --projector-checkpoint "$warm" --target v --rank 8 --last-n-layers 8 \
    --alpha 8 --epochs 10 --max-pixels 100352 "${extra[@]}"
done

# Online evaluation intentionally uses unseen seeds and a fixed alpha chosen
# before comparing objectives. See results/gonogo_current.json for summaries.
for objective in warm return shuffle ce; do
  checkpoint="$warm"
  case "$objective" in
    return) checkpoint="$out/miniwob_return_v_r8_l8_a8_e10/kv_projectors.pt" ;;
    shuffle) checkpoint="$out/miniwob_shuffle_v_r8_l8_a8_e10/kv_projectors.pt" ;;
    ce) checkpoint="$out/miniwob_ce_v_r8_l8_a8_e10/kv_projectors.pt" ;;
  esac
  .venv/bin/python scripts/evaluate_miniwob_online.py --tasks click-color \
    --seeds {601..630} --checkpoint "$checkpoint" --target v --rank 8 \
    --last-n-layers 8 --alpha 16 --max-pixels 100352 \
    --output "$out/miniwob_color_${objective}_a16_s601_630.json"
done
