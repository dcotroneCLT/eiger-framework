#!/usr/bin/env bash
# Generate a fully pinned, hash-verified lockfile for THIS machine's platform.
#
# Output: requirements/lock-<os>-<arch>.txt   (e.g. lock-darwin-arm64.txt)
#
# Why one lock per platform: PyTorch and some numeric wheels differ between
# macOS/Linux and arm64/x86_64, so a single file cannot serve every machine.
# Machines with the same <os>-<arch> share the same lock and must produce the
# same packages. Commit the lock files; never regenerate one silently.
#
# Usage:  scripts/lock_env.sh            (regenerates the lock for this platform)
#         scripts/lock_env.sh --check    (fails if the current lock is stale)
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
LOCK="requirements/lock-${OS}-${ARCH}.txt"
TMPVENV=".venv-lock"

"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  || { echo "ERROR: lock must be generated with Python 3.12 (found $($PY --version))."; exit 1; }

rm -rf "$TMPVENV"
"$PY" -m venv "$TMPVENV"
# shellcheck disable=SC1091
source "$TMPVENV/bin/activate"
python -m pip install --quiet --upgrade pip pip-tools

mkdir -p requirements
# Resolve the project's own dependency ranges (pyproject.toml), including
# dev and data-import extras, to exact versions with SHA-256 hashes.
pip-compile --quiet --generate-hashes --strip-extras \
  --extra dev --extra data-import \
  --output-file "$LOCK" pyproject.toml

deactivate
rm -rf "$TMPVENV"

if [[ "${1:-}" == "--check" ]]; then
  echo "Lock check: re-resolving to verify it is up to date..."
  TMP="$(mktemp)"
  "$PY" -m venv "$TMPVENV" && source "$TMPVENV/bin/activate" \
    && python -m pip install --quiet pip-tools \
    && pip-compile --quiet --generate-hashes --strip-extras --extra dev --extra data-import \
         --output-file "$TMP" pyproject.toml
  deactivate || true
  rm -rf "$TMPVENV"
  if diff -q "$LOCK" "$TMP" >/dev/null; then
    echo "OK: $LOCK is up to date."
  else
    echo "STALE: $LOCK differs from a fresh resolution. Regenerate and review the diff."
    diff "$LOCK" "$TMP" | head -40 || true
    exit 1
  fi
  rm -f "$TMP"
fi

echo "Lock written: $LOCK"
echo "Next: commit it, then run scripts/env_fingerprint.py --out fingerprint_$(hostname -s).json"
