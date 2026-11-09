#!/usr/bin/env bash
# Experiment 2: synthetic long-range recall, 4 variants x 2 seeds.
set -e
PY=${PY:-python}
cd "$(dirname "$0")/.."
for seed in 0 1; do
  for attn in mha swa mla mla_swa; do
    $PY recall.py --attn=$attn --seed=$seed
  done
done
# Small window: receptive field n_layer*(W-1) = 60 < most distances -> shows the hard limit
for attn in swa mla_swa; do
  $PY recall.py --attn=$attn --seed=0 --window=16 --out=results/recall_w16
done
