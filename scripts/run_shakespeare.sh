#!/usr/bin/env bash
# Experiment 1: char-level Tiny Shakespeare, 4 attention variants (+ vanilla nanoGPT) x 3 seeds.
# Usage: PY=python bash scripts/run_shakespeare.sh
set -e
PY=${PY:-python}
cd "$(dirname "$0")/.."
$PY data/shakespeare_char/prepare.py
for seed in 0 1 2; do
  # vanilla nanoGPT reference (learned absolute positions, standard attention)
  $PY train.py config/shakespeare_hybrid.py --attn=mha --pos_emb=learned --seed=$seed \
      --out_dir=out/shakespeare/nanogpt_s$seed --metrics_file=results/shakespeare/nanogpt_s$seed.json
  for attn in mha swa mla mla_swa; do
    $PY train.py config/shakespeare_hybrid.py --attn=$attn --seed=$seed \
        --out_dir=out/shakespeare/${attn}_s$seed --metrics_file=results/shakespeare/${attn}_s$seed.json
  done
done
