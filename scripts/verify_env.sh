#!/usr/bin/env bash
# Compare this machine's environment with the committed reference fingerprint.
#
# Usage: scripts/verify_env.sh
#
# Writes reference/fingerprint_<host>.json for this machine and compares it with
# reference/fingerprint_reference.json. Exit code 0 = identical; 1 = differences.
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="$(hostname -s 2>/dev/null || echo host)"
REF="reference/fingerprint_reference.json"
OUT="reference/fingerprint_${HOST}.json"

mkdir -p reference
python3 scripts/env_fingerprint.py --out "$OUT"

if [[ ! -f "$REF" ]]; then
  echo "No reference yet. Copy $OUT to $REF on the agreed reference machine"
  echo "(or make this one the reference after a successful baseline parity check)."
  exit 0
fi

python3 scripts/env_fingerprint.py --compare "$REF" "$OUT"
