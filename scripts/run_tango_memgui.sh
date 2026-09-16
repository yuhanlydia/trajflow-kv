#!/usr/bin/env bash
# Invoke the OFFICIAL MemGUI runtime; it owns initialization, actions and judging.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MEMGUI_DIR:?Set MEMGUI_DIR to an installed lgy0404/MemGUI-Bench checkout}"
: "${TANGO_LOG_ROOT:?Set a fresh absolute TANGO_LOG_ROOT for this method/seed}"
PORT="${TANGO_PORT:-8127}"
MODEL="${TANGO_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
TASKS="${TANGO_TASKS:-ALL}"
PYTHON="${TANGO_PYTHON:-python}"
ARGS=(--model "$MODEL" --port "$PORT" --seed "${TANGO_SEED:-0}")
if [[ -n "${TANGO_CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$TANGO_CHECKPOINT"); fi
if [[ -n "${TANGO_ADAPTER:-}" ]]; then ARGS+=(--adapter "$TANGO_ADAPTER"); fi
# This script does NOT start privileged containers or change judge credentials.
# Start official healthy backends first: sudo uv run mg env check; sudo uv run mg env run.
mkdir -p "$TANGO_LOG_ROOT"
cd "$ROOT"
"$PYTHON" -m tango_iclr.serve "${ARGS[@]}" > "$TANGO_LOG_ROOT/policy_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
"$PYTHON" - "$PORT" "$SERVER_PID" <<'PY'
import os,sys,time,urllib.request
port,pid=int(sys.argv[1]),int(sys.argv[2])
for _ in range(600):
    os.kill(pid,0)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:
            if r.status==200:break
    except Exception:time.sleep(1)
else:raise SystemExit('Policy endpoint startup failed; inspect server log')
PY
cd "$MEMGUI_DIR"
uv run mg eval --agent-type qwen3vl --model-name tango-gui \
  --llm-base-url "http://127.0.0.1:$PORT/v1" --api-key local-not-used \
  --tasks "$TASKS" --pass-at-k 1 --max-concurrency 1 --llm-max-concurrency 1 \
  --log-file-root "$TANGO_LOG_ROOT"
