#!/usr/bin/env bash
# Reproduce every number in the presentation, from the raw data.
# Usage: bash run_all.sh [path/to/openoximetry/folder]
#
# Step 2 needs PhysioNet access to the OpenOximetry Repository (restricted
# license, data not redistributed). Without it, the committed aggregate file
# results/openox_error_model.json is used and every later step still runs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
export PYTHONPATH=src PYTHONWARNINGS=ignore

echo "== Step 1: label audit (NEWS2 vs dataset labels) =="
python src/label_audit.py

if [ $# -ge 1 ]; then
  echo "== Step 2: fit oximeter error model from OpenOximetry =="
  (cd results && python ../src/openox_error_model.py "$1")
else
  echo "== Step 2: skipped (no OpenOximetry path given); using results/openox_error_model.json =="
fi

echo "== Step 3: risk models vs baselines on held-out test set =="
python src/models.py

echo "== Step 4: oximeter-bias stress test =="
python src/stress_test.py

echo "== Step 5: where the bias bites =="
python src/where_bias_bites.py

echo "== Step 6: per-patient linked table (Health Risk x OpenOximetry) =="
python src/linked_table.py

echo "== Step 7: tipping-point analysis =="
python src/tipping_point.py

echo "== Step 8: subgroup analysis and failure examples =="
python src/subgroups.py

echo "== Step 9: build interactive ward view (docs/index.html) =="
python src/build_ward_view.py
