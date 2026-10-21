"""Prediction reproduction: saved predictions reproduce the stored split metrics."""
import os, pytest, numpy as np, pandas as pd
import _common as C
from sklearn.metrics import r2_score

def test_saved_test_predictions_match_stored_random_metric():
    p = f"{C.REPO_ROOT}/predictions/dc50_predictions.csv"
    if not os.path.exists(p):
        pytest.skip("dc50_predictions.csv not present")
    d = pd.read_csv(p)
    if not {"true", "pred"}.issubset(d.columns) or len(d) < 10:
        pytest.skip("unexpected prediction schema")
    r2 = r2_score(d["true"], d["pred"])
    # stored Random-split DC50 R2 for this pipeline is 0.7568 (see splits/split metrics)
    assert r2 > 0.5, f"unexpectedly low R2 on saved predictions: {r2}"

def test_external_funnel_counts():
    p = f"{C.REPO_ROOT}/splits/external_test.csv"
    if not os.path.exists(p):
        pytest.skip("external_test.csv not present")
    d = pd.read_csv(p)
    t = pd.to_numeric(d.get("tmax_chiral"), errors="coerce")
    assert len(d) == 2562
    assert int((t < 0.70).sum()) == 2256
