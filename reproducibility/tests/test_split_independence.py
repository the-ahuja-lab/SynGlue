"""Split independence: scaffold/E3 must be compound-disjoint; external overlap must be zero."""
import os, pytest, pandas as pd
import _common as C

def _report():
    p = os.path.join(C.RESULTS, "leakage_report.csv")
    if not os.path.exists(p):
        pytest.skip("run scripts/06_check_data_leakage.py first")
    return pd.read_csv(p).set_index("split")

def test_scaffold_and_e3_are_compound_disjoint():
    d = _report()
    for s in ["Scaffold-held", "E3-held (VHL)"]:
        if s in d.index:
            assert d.loc[s, "shared_compound_ids_train_test"] == 0

def test_random_and_target_are_row_level():
    d = _report()
    for s in ["Random", "Target-held"]:
        if s in d.index:
            assert d.loc[s, "shared_compound_ids_train_test"] > 0

def test_no_external_overlap_in_any_split():
    d = _report()
    for s in d.index:
        assert d.loc[s, "test_overlap_external"] == 0
        assert d.loc[s, "train_overlap_external"] == 0
