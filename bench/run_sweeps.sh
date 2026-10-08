#!/bin/bash
# Sequential engine sweeps on NYUv2-13 (100 held-out images). Results: bench/*.json, log: bench/sweeps.log
# Prepare the data first: python scripts/prepare_nyu.py   (pip install -e ".[bench]")
# Note: the published JSON files were produced before the held-out set was moved to the end of the
# permutation (see docs/BENCHMARK.md), so re-running gives slightly different numbers.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
B="scripts/bench_engine.py --test 100"
for m in small base large huge_plus; do
  $PY $B --model $m --refs 5 10 20 40 --modes knn cluster hybrid --refines bilinear --out bench/model_$m.json
done
$PY $B --model large --refs 10 --svd-frac 0.0 --modes knn hybrid --refines bilinear --out bench/svd_0.json
$PY $B --model large --refs 10 --svd-frac 0.25 --modes knn hybrid --refines bilinear --out bench/svd_25.json
$PY $B --model large --refs 10 --svd-frac 0.75 --modes knn hybrid --refines bilinear --out bench/svd_75.json
$PY $B --model large --refs 10 --k 5 10 20 40 --tau 0.5 0.6 0.7 --modes cluster hybrid --refines bilinear --out bench/k_tau.json
$PY $B --model large --refs 40 --compress 1.0 0.5 0.2 0.1 0.05 --modes knn hybrid --refines bilinear --out bench/compress.json
$PY $B --model large --image-size 768 --refs 10 --modes knn hybrid --refines bilinear superpixel --out bench/size_768.json
$PY $B --model large --image-size 1280 --refs 10 --modes knn hybrid --refines bilinear superpixel --out bench/size_1280.json
$PY $B --model large --refs 10 40 --modes knn smooth hybrid smooth_insid3 --refines bilinear superpixel --out bench/smooth.json
$PY $B --model large --refs 10 --modes knn smooth hybrid --refines bilinear superpixel --out bench/baseline_refs10.json
for s in 1 2; do
  $PY $B --model large --refs 40 --seed $s --compress 1.0 0.5 0.2 0.1 --modes knn hybrid --refines bilinear --out bench/compress_seed$s.json
done
echo SWEEPS_DONE
