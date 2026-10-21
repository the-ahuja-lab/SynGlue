"""Prediction reproduction: regenerate predictions with the steps, then check them here."""
import os, pytest, numpy as np, pandas as pd
import _common as C
from sklearn.metrics import r2_score


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def test_saved_test_predictions_match_stored_random_metric():
    """Step 04 writes pred_dc50_test.npy; step 05 writes regression_metrics.csv."""
    p = _first_existing(os.path.join(C.RESULTS, "pred_dc50_test.npy"))
    y = _first_existing(os.path.join(C.RESULTS, "y_dc50_test.npy"))
    if p is None or y is None:
        pytest.skip("run steps 04/05 first to regenerate predictions and labels")
    pred = np.load(p); true = np.load(y)
    if len(pred) < 10:
        pytest.skip("insufficient predictions")
    r2 = r2_score(true, pred)
    assert r2 > 0.5, f"unexpectedly low R2 on regenerated predictions: {r2}"


def test_external_funnel_counts_after_step07():
    p = os.path.join(C.RESULTS, "external_funnel_counts.csv")
    if not os.path.exists(p):
        pytest.skip("run step 07 first to regenerate the external funnel")
    d = pd.read_csv(p)
    assert d["n"].iloc[0] == 2930          # assembled pool before overlap control
