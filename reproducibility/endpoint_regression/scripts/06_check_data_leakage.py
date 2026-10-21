#!/usr/bin/env python3
"""06_check_data_leakage.py  [P0]

Check exact compound overlap across Transformer training, RF fitting and test partitions, and
between the training population and the external benchmark.

What it does
  1. Rebuilds the row-level mapped order (9,252 rows) from the curated table + component libraries.
  2. Loads the saved split indices (four_split_indices.npz).
  3. For each split, reports: rows and unique compounds in train/test, shared compound IDs,
     test rows sharing a compound with training, canonical-SMILES overlap, and overlap with the
     external benchmark (both directions).
  4. Checks the v4 "extras" files for external compounds merged into training
     (metadata_{dc50,dmax}_extra.csv) and reports unique-compound and row counts.

Outputs (results/)
  mapped_compound_order.csv        compound_id, smiles, component ids, canonical SMILES
  leakage_report.csv               per-split independence table
  leakage_v4_extras.json           external compounds present in the v4 retrain
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

SPLITS = ["Random", "Scaffold-held", "Target-held", "E3-held (VHL)"]


def canon(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def rebuild_mapped_order():
    p = pd.read_csv(C.require(C.PATHS["training_table"]), low_memory=False)
    wh = pd.read_csv(C.require(C.PATHS["warhead_table"]), low_memory=False)
    lk = pd.read_csv(C.require(C.PATHS["linker_table"]), low_memory=False)
    e3 = pd.read_csv(C.require(C.PATHS["e3_table"]), low_memory=False)
    for df, cols in [(p, ["warhead_Uniprot", "warhead_Target", "E3 ligase", "Linker ID",
                          "PROTAC_Compound_ID"]),
                     (wh, ["warhead_Uniprot", "warhead_Target"]),
                     (lk, ["linker_Compound_ID"]), (e3, ["E3_Ligase"])]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.upper()
    rows = []
    for _, r in p.iterrows():
        w = wh[(wh["warhead_Uniprot"] == r["warhead_Uniprot"]) |
               (wh["warhead_Target"] == r["warhead_Target"])]
        l = lk[lk["linker_Compound_ID"] == r["Linker ID"]]
        e = e3[e3["E3_Ligase"] == r["E3 ligase"]]
        if w.empty or l.empty or e.empty:
            continue
        rows.append(dict(compound_id=str(r["PROTAC_Compound_ID"]), smiles=str(r["PROTAC_SMILES"]),
                         warhead_id=str(w.iloc[0]["warhead_Compound_ID"]),
                         linker_id=str(r["Linker ID"]), e3_id=str(e.iloc[0]["E3_Compound_ID"])))
    m = pd.DataFrame(rows)
    m["canon"] = [canon(s) for s in m["smiles"]]
    C.save_csv(m, "mapped_compound_order.csv")
    print(f"  mapped rows: {len(m)} (expected 9,252 for the compound-level pipeline)")
    return m


def external_overlap(m):
    ext = pd.read_csv(C.require(C.PATHS["external_pool"]), low_memory=False)
    return set(c for c in map(canon, ext["SMILES"].dropna().astype(str)) if c)


def v4_extras_check():
    out = {}
    for tag in ["dc50", "dmax"]:
        p = os.path.join(C.ROOT, "revision", "benchmarking", "generalization_data_v4",
                         f"metadata_{tag}_extra.csv")
        base = os.path.join(C.ROOT, "revision", "benchmarking", "generalization_data_v4",
                            f"metadata_{tag}.csv")
        if not (os.path.exists(p) and os.path.exists(base)):
            out[tag] = "files not found"
            continue
        ext = set(c for c in map(canon, pd.read_csv(C.PATHS["external_pool"])["SMILES"]
                                 .dropna().astype(str)) if c)
        for name, path in [("extra", p), ("main", base)]:
            d = pd.read_csv(path, low_memory=False)
            col = "canon" if "canon" in d.columns else ("SMILES" if "SMILES" in d.columns else None)
            if col is None:
                continue
            cset = [canon(s) for s in d[col].astype(str)]
            hit = [c for c in cset if c in ext]
            out.setdefault(tag, {})[name] = dict(rows=len(d), overlapping_rows=len(hit),
                                                 overlapping_unique_compounds=len(set(hit)))
    return out


def main():
    print("rebuilding mapped order...")
    m = rebuild_mapped_order()
    ext_canon = external_overlap(m)
    print(f"  external benchmark compounds: {len(ext_canon)}")

    rows = []
    if os.path.exists(C.PATHS["split_indices"]):
        z = np.load(C.PATHS["split_indices"], allow_pickle=True)
        for split in SPLITS:
            if f"tr_{split}" not in z:
                continue
            tr, te = z[f"tr_{split}"], z[f"te_{split}"]
            tr_comp = set(m["compound_id"].iloc[tr]); te_comp = set(m["compound_id"].iloc[te])
            tr_can = set(m["canon"].iloc[tr].dropna()); te_can = set(m["canon"].iloc[te].dropna())
            rows.append(dict(
                split=split, n_train=len(tr), n_test=len(te),
                train_unique_compounds=len(tr_can), test_unique_compounds=len(te_can),
                shared_compound_ids_train_test=len(tr_comp & te_comp),
                test_rows_sharing_compound_with_train=int(
                    m["compound_id"].iloc[te].isin(tr_comp).sum()),
                canon_overlap_train_test=len(tr_can & te_can),
                test_overlap_external=len(te_can & ext_canon),
                train_overlap_external=len(tr_can & ext_canon)))
    else:
        print("  split indices not found; reporting external overlap only")
    rep = pd.DataFrame(rows)
    if len(rep):
        C.save_csv(rep, "leakage_report.csv")
        print(rep.to_string(index=False))

    C.save_json(dict(v4_extras=v4_extras_check(),
                     note=("random and target splits are row-level and share compounds; "
                           "scaffold and E3 splits are compound-disjoint")),
                "leakage_v4_extras.json")


if __name__ == "__main__":
    main()
