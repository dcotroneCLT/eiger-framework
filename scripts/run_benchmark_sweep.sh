#!/usr/bin/env bash
# Runs all generated paper-facing sweep configs, one log per run.
# Uses this machine's venv explicitly, so it does not depend on the
# terminal's active environment. Stops at the first failure (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="venv-$(hostname -s)/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found. Run scripts/install_locked.sh first."; exit 1; }
"$PY" -c "import eiger" || { echo "ERROR: eiger not importable with $PY"; exit 1; }
mkdir -p logs
for f in experiments/poison_rate_sweep_benchmark/*.yaml; do
  name=$(basename "$f" .yaml)
  echo "=== $name $(date +%H:%M:%S) ==="
  if ! "$PY" -m eiger run "$f" > "logs/$name.log" 2>&1; then
    echo "FAILED: $name (see logs/$name.log)"; tail -5 "logs/$name.log"; exit 1
  fi
done
echo "sweep completo: $(date +%H:%M:%S)"
