#!/usr/bin/env bash
# Runs every config in a sweep directory, one log per run.
# Usage: scripts/run_benchmark_sweep.sh [config_dir]
#   default config_dir: experiments/poison_rate_sweep_benchmark
# Uses this machine's venv explicitly, so it does not depend on the
# terminal's active environment. Stops at the first failure (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG_DIR="${1:-experiments/poison_rate_sweep_benchmark}"
# Prefer the active virtualenv: macOS can change `hostname -s` when the
# network changes, which would point to a venv that does not exist.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then PY="$VIRTUAL_ENV/bin/python"; else PY="venv-$(hostname -s)/bin/python"; fi
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Run scripts/install_locked.sh first."; exit 1; }
"$PY" -c "import eiger" || { echo "ERROR: eiger not importable with $PY"; exit 1; }
[[ -d "$CFG_DIR" ]] || { echo "ERROR: $CFG_DIR not found."; exit 1; }
LOG_DIR="logs/$(basename "$CFG_DIR")"
mkdir -p "$LOG_DIR"
for f in "$CFG_DIR"/*.yaml; do
  name=$(basename "$f" .yaml)
  echo "=== $name $(date +%H:%M:%S) ==="
  if ! "$PY" -m eiger run "$f" > "$LOG_DIR/$name.log" 2>&1; then
    echo "FAILED: $name (see $LOG_DIR/$name.log)"; tail -5 "$LOG_DIR/$name.log"; exit 1
  fi
done
echo "sweep completo: $(date +%H:%M:%S)"
