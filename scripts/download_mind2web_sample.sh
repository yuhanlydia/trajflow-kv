#!/usr/bin/env bash
set -euo pipefail

# Smallest official public training shard (CC BY 4.0). Test archives are not
# downloaded because the dataset authors prohibit redistribution after unzip.
destination="${1:-data/mind2web_raw/train_10.json}"
mkdir -p "$(dirname "$destination")"
curl -L --fail --show-error \
  'https://huggingface.co/datasets/osunlp/Mind2Web/resolve/main/data/train/train_10.json?download=true' \
  -o "$destination"
echo "downloaded $destination"
