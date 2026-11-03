#!/usr/bin/env bash
# Install the exact, hash-verified dependency set for this machine's platform.
#
# Usage: scripts/install_locked.sh
#
# - Picks requirements/lock-<os>-<arch>.txt for this machine.
# - Refuses to run on another Python version than the one the lock was made with.
# - Installs with --require-hashes: any package whose hash differs is rejected.
# - Installs the project with --no-deps so nothing is re-resolved.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
LOCK="requirements/lock-${OS}-${ARCH}.txt"
VENV="venv-$(hostname -s 2>/dev/null || echo host)"

if [[ ! -f "$LOCK" ]]; then
  echo "ERROR: no lockfile for ${OS}-${ARCH} ($LOCK)."
  echo "Generate it on this machine with scripts/lock_env.sh, review it, and commit it."
  exit 1
fi

"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || { echo "ERROR: Python 3.12 required (found $($PY --version))."; exit 1; }

if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --require-hashes -r "$LOCK"
python -m pip install --no-deps -e .

echo "Installed from $LOCK into $VENV."
echo "Activate with: source $VENV/bin/activate"
echo "Then verify: scripts/verify_env.sh"
