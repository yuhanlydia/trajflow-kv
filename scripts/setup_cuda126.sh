#!/usr/bin/env bash
set -euo pipefail

# RTX A4000 host driver in the reference environment supports CUDA 12.7, so
# use CUDA 12.6 wheels instead of PyPI's newest CUDA build.
.venv/bin/pip install --force-reinstall 'torch==2.7.1' \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install 'torchvision==0.22.1' \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
