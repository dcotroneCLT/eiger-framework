#!/usr/bin/env bash
# Runs all 5 points of the poison-rate sweep (0/1/3/5/10%) in sequence.
#
# Prerequisites (see docs/REPRODUCING.md / Makefile):
#   - `make bootstrap` once (installs Docker + native Ollama, pulls the model)
#   - `make up` before this script (starts Qdrant, checks Ollama is reachable)
#
# Each of the 5 configs uses a distinct `collection_name` and `output_dir`,
# so runs do not overwrite each other's Qdrant collection or results.json.
# A failure on one point stops the script (set -e) rather than silently
# continuing with a partial sweep.
#
# Usage:
#   ./scripts/run_poison_rate_sweep.sh

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
  "experiments/poison_rate_sweep/sweep_0pct.yaml"
  "experiments/poison_rate_sweep/sweep_1pct.yaml"
  "experiments/poison_rate_sweep/sweep_3pct.yaml"
  "experiments/poison_rate_sweep/sweep_5pct.yaml"
  "experiments/poison_rate_sweep/sweep_10pct.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running: ${cfg} ==="
  python3 -m eiger run "${cfg}"
  echo "=== Done: ${cfg} ==="
  echo
done

echo "All 5 sweep points complete. Results under results/poison_rate_sweep/."
echo "Aggregate into the Excel matrix with:"
echo "  python scripts/build_results_matrix.py"
