#!/usr/bin/env python3
"""Interpretability — Part 2: component-ablation deltas via frozen RFs (rf_predict env)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
OUT = os.path.join(TIR, "interp")
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"

Zf = np.load(os.path.join(OUT, "Z_full.npy"))
rf50 = joblib.load(os.path.join(SYNGLUE, "models", "rf_dc50.joblib"))
rfmax = joblib.load(os.path.join(SYNGLUE, "models", "rf_dmax.joblib"))

pf50 = rf50.predict(Zf)
pfmax = rfmax.predict(Zf)

rows = []
for tag in ["warhead", "linker", "e3"]:
    Za = np.load(os.path.join(OUT, f"Z_abl_{tag}.npy"))
    pa50 = rf50.predict(Za)
    pamax = rfmax.predict(Za)
    d50 = np.abs(pf50 - pa50)          # |Δ log10 DC50|
    dmax = np.abs(pfmax - pamax)       # |Δ Dmax %|
    rows.append(dict(component=tag,
                     mean_abs_dlog10_dc50=float(d50.mean()),
                     median_abs_dlog10_dc50=float(np.median(d50)),
                     pct_dc50_changed_gt0_1=float(np.mean(d50 > 0.1)),
                     mean_abs_dmax=float(dmax.mean()),
                     median_abs_dmax=float(np.median(dmax)),
                     pct_dmax_changed_gt2=float(np.mean(dmax > 2.0))))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ablation_results.csv"), index=False)
print(df.to_string(index=False))
print("\nInterpretation: larger delta = stronger dependence on that component.")
