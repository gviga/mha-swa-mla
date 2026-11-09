#!/usr/bin/env bash
# Reproduce everything: tests, 4 experiments, figures + summary tables.
set -e
PY=${PY:-python}
cd "$(dirname "$0")/.."
$PY -m pytest -q tests
PY=$PY bash scripts/run_shakespeare.sh
PY=$PY bash scripts/run_recall.sh
$PY bench_attention.py
$PY probe.py
$PY plots.py
$PY plots_probe.py
