#!/usr/bin/env python3
"""
Export all 4-split compound assignments for DC50 and Dmax tasks with reported values.

For each task (dc50, dmax) and each split (random, scaffold_held_out, target_held_out, e3_held_out):
  - every training compound with its split role (train/val/test)
  - reported DC50 (nM) / Dmax (%) values
  - scaffold, target, E3 annotations

Output: revision/benchmarking/generalization_data_v4/split_assignments/
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
BASE = "/storage/savi/saveenas/Projects/SynGlue"
REV = f"{BASE}/revision"
DATA = f"{REV}/benchmarking/generalization_data_v4"
OUT = f"{DATA}/split_assignments"
os.makedirs(OUT, exist_ok=True)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

RANDOM_STATE = 42
SPLITS = ["random", "scaffold_held_out", "target_held_out", "e3_held_out"]

def load_task(task):
    X = np.load(f"{DATA}/X_{task}_pca.npy")
    y = np.load(f"{DATA}/y_{task}.npy")
    meta = pd.read_csv(f"{DATA}/metadata_{task}.csv")
    ex = {'dc50': ('X_dc50_extra_full.npy','y_dc50_extra.npy','metadata_dc50_extra.csv'),
          'dmax': ('X_dmax_extra_full.npy','y_dmax_extra.npy','metadata_dmax_extra.csv'),
          'dmax90': (None,None,None)}[task]
    if ex[0] and os.path.exists(f"{DATA}/{ex[0]}"):
        pca = joblib.load(f"{DATA}/pca_512.joblib")
        Xe = np.load(f"{DATA}/{ex[0]}")
        Xe_p = pca.transform(Xe.reshape(-1,4800)).reshape(Xe.shape[0],3,512).astype(np.float32)
        ye = np.load(f"{DATA}/{ex[1]}")
        me = pd.read_csv(f"{DATA}/{ex[2]}")
        X = np.concatenate([X, Xe_p]); y = np.concatenate([y, ye])
        meta = pd.concat([meta, me], ignore_index=True)
    return X, y, meta

def murcko(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) or None
    except: return None

def split_functions(meta):
    def random_split():
        idx = np.arange(len(meta))
        tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
        tr, va = train_test_split(tr, test_size=0.15/0.8, random_state=RANDOM_STATE)
        return tr, va, te
    def target_held_out():
        targets = meta['target_uniprot'].value_counts()
        rng = np.random.RandomState(RANDOM_STATE)
        cand = [t for t in targets.index if str(t) != 'NAN' and targets[t] >= 5]
        n_hold = max(1, int(len(cand) * 0.2))
        held = set(rng.choice(cand, size=n_hold, replace=False))
        te = np.where(meta['target_uniprot'].isin(held))[0]
        trva = np.where(~meta['target_uniprot'].isin(held))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    def e3_held_out():
        te = np.where(meta['e3'].isin(['VHL']))[0]
        trva = np.where(~meta['e3'].isin(['VHL']))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    def scaffold_held_out():
        sc_counts = meta['scaffold'].value_counts()
        rng = np.random.RandomState(RANDOM_STATE)
        cand = [s for s in sc_counts.index if pd.notna(s) and sc_counts[s] >= 3]
        n_hold = max(1, int(len(cand) * 0.2))
        held = set(rng.choice(cand, size=n_hold, replace=False))
        te = np.where(meta['scaffold'].isin(held))[0]
        trva = np.where(~meta['scaffold'].isin(held))[0]
        tr, va = train_test_split(trva, test_size=0.15, random_state=RANDOM_STATE)
        return tr, va, te
    return {"random": random_split, "scaffold_held_out": scaffold_held_out,
            "target_held_out": target_held_out, "e3_held_out": e3_held_out}

def main():
    summary_rows = []
    for task in ['dc50', 'dmax']:
        X, y, meta = load_task(task)
        if 'scaffold' not in meta.columns:
            meta['scaffold'] = meta['SMILES'].apply(murcko)
        n = len(meta)
        print(f"\n{'='*75}\nTASK {task.upper()} — n={n}\n{'='*75}")

        fns = split_functions(meta)
        role_cols = {}
        for split_name, fn in fns.items():
            tr, va, te = fn()
            roles = pd.Series('train', index=range(n))
            roles[va] = 'val'
            roles[te] = 'test'
            role_cols[split_name] = roles

        # Build per-compound export
        export = meta.copy()
        export['canon'] = export['SMILES'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) if Chem.MolFromSmiles(s) else None)
        export['y_value'] = y  # log10 DC50 for dc50; % for dmax
        # reported value (original units)
        if task == 'dc50':
            export['Reported_DC50_nM'] = 10 ** y
            export['Value_Type'] = 'DC50'
        else:
            export['Reported_Dmax_pct'] = y
            export['Value_Type'] = 'Dmax'

        for split_name in SPLITS:
            export[f'split_{split_name}'] = role_cols[split_name].values

        out_file = f"{OUT}/split_assignments_{task}.csv"
        export.to_csv(out_file, index=False)

        # Counts summary
        print(f"\nSplit counts ({task}):")
        for split_name in SPLITS:
            r = export[f'split_{split_name}']
            n_tr = (r == 'train').sum(); n_va = (r == 'val').sum(); n_te = (r == 'test').sum()
            test_vals = export[r == 'test']
            if task == 'dc50':
                med = test_vals['Reported_DC50_nM'].median() if len(test_vals) else np.nan
                print(f"  {split_name:18s}: train={n_tr:4d} val={n_va:4d} test={n_te:4d} | test DC50 median={med:.1f} nM")
            else:
                med = test_vals['Reported_Dmax_pct'].median() if len(test_vals) else np.nan
                print(f"  {split_name:18s}: train={n_tr:4d} val={n_va:4d} test={n_te:4d} | test Dmax median={med:.1f}%")
            summary_rows.append({"task": task, "split": split_name, "n_train": n_tr, "n_val": n_va,
                                 "n_test": n_te, "test_median": med})

        # per-test-compound table for each split
        for split_name in SPLITS:
            test_df = export[export[f'split_{split_name}'] == 'test'][
                ['SMILES','canon','target_uniprot','target_name','e3','scaffold',
                 'Reported_DC50_nM' if task=='dc50' else 'Reported_Dmax_pct']]
            test_df.to_csv(f"{OUT}/{task}_{split_name}_test_compounds.csv", index=False)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(f"{OUT}/split_counts_summary.csv", index=False)
    print(f"\nSaved all to {OUT}")
    print(summ.to_string(index=False))

if __name__ == '__main__':
    main()
