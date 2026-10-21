#!/usr/bin/env bash
# run_all.sh — execute the light reproduction steps in order (heavy training steps are manual).
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
run() { echo "== $* =="; "$@" || echo "  (skipped: missing input — see the step README)"; }

run $PY endpoint_regression/scripts/06_check_data_leakage.py
run $PY endpoint_regression/scripts/05_evaluate_regression.py
run $PY external_benchmark/scripts/07_evaluate_external.py
run $PY linker_multiclass/scripts/08_train_linker_classifier.py --mode grouped
run $PY figures/scripts/09_reproduce_figures.py

cat <<'TXT'

Heavy stages (run manually when data + GROVER are available):
  endpoint_regression/scripts/01_prepare_components.py
  endpoint_regression/scripts/02_train_multitask_transformer.py --tensor X.npy --dc50 y50.npy --dmax ydm.npy
  endpoint_regression/scripts/03_extract_embeddings.py --mode verify|recompute
  endpoint_regression/scripts/04_train_rf_regressors.py
TXT
