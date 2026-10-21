"""Metric implementations and orientation conventions."""
import numpy as np, pytest
from sklearn.metrics import roc_auc_score

def test_orientation_rule_is_apriori():
    """Regressors must be oriented as potency = -log10(DC50 nM); classifiers use P(active)."""
    dc50 = np.array([1.0, 10.0, 100.0, 1000.0])
    potency = -np.log10(dc50)
    assert np.all(np.diff(potency) < 0)          # lower DC50 -> higher potency score

def test_auroc_bounds_and_degenerate_guard():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.5 <= roc_auc_score(y, s) <= 1.0
    # a single-class target is undefined: implementations either raise or return NaN
    try:
        v = roc_auc_score(np.zeros(4), s)
        assert np.isnan(v), f"expected NaN for single-class target, got {v}"
    except ValueError:
        pass
