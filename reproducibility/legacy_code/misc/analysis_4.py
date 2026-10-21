#!/usr/bin/env python3
"""
Analysis 3 (R1.2) — isomeric PROTAC series: same warhead + same E3, linker varies
→ potency varies. Supports: 'optimal warhead alone does not make the best PROTAC'.
Data: training table (Upadted_PROTAC_features.csv).
Series = (warhead_Target, E3 ligase) with >=2 distinct linkers and >=2 DC50 values.
Outputs: outputs/isomer_series.csv, outputs/isomer_plot.png/.svg
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
DATA = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{DATA}/Upadted_PROTAC_features.csv", low_memory=False,
                 usecols=lambda c: not c.startswith("Grover_"))
for c in ["warhead_Target", "E3 ligase", "Linker ID", "PROTAC_Compound_ID"]:
    df[c] = df[c].astype(str).str.strip().str.upper()
df["DC50 (nM)"] = pd.to_numeric(df["DC50 (nM)"], errors="coerce")
print(f"rows: {len(df)}")

# per-series: (target, E3) with >=2 distinct linkers and >=2 DC50 values
rows = []
for (tgt, e3), g in df.groupby(["warhead_Target", "E3 ligase"]):
    gv = g.dropna(subset=["DC50 (nM)"])
    n_link = g["Linker ID"].nunique()
    n_dc50 = gv["DC50 (nM)"].nunique()
    if n_link >= 2 and n_dc50 >= 2:
        vals = gv["DC50 (nM)"].values
        rows.append(dict(target=tgt, e3=e3, n_compounds=g["PROTAC_Compound_ID"].nunique(),
                         n_linkers=n_link, n_dc50=n_dc50,
                         dc50_min=vals.min(), dc50_max=vals.max(),
                         fold_range=vals.max() / max(vals.min(), 1e-3)))

ser = pd.DataFrame(rows)
ser = ser.sort_values("fold_range", ascending=False).reset_index(drop=True)
ser.to_csv(os.path.join(OUT, "isomer_series.csv"), index=False)

# BRD4 series specifically (same warhead, different linkers, cross-cell-line) — from training table
brd4 = df[df["warhead_Target"] == "BRD4"].dropna(subset=["DC50 (nM)"])
print(f"\nSeries (target,E3) with >=2 linkers & >=2 DC50: {len(ser)}")
print(f"Series with >10-fold DC50 spread: {(ser['fold_range'] > 10).sum()}")
print(f"Median within-series fold range: {ser['fold_range'].median():.1f}×")
print(f"Max within-series fold range: {ser['fold_range'].max():.1f}×")
if len(brd4):
    print(f"\nBRD4 rows with DC50: {len(brd4)} | distinct linkers: {brd4['Linker ID'].nunique()} | "
          f"DC50 range: {brd4['DC50 (nM)'].min():.1f}-{brd4['DC50 (nM)'].max():.1f} nM")

# ─── Plot ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.hist(np.log10(ser["fold_range"]), bins=20, color="#1f77b4", alpha=0.85)
ax.axvline(np.log10(10), color="r", ls="--", lw=1, label="10×")
ax.set_xlabel("log10(within-series DC50 fold range)")
ax.set_ylabel("Series count")
ax.set_title(f"Isomeric PROTAC series (n={len(ser)})")
ax.legend()
ax = axes[1]
top = ser.head(10)
ax.barh([f"{t}/{e}" for t, e in zip(top['target'], top['e3'])], top["fold_range"], color="#d62728", alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("DC50 fold range (max/min)")
ax.set_title("Top 10 series by potency spread")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "isomer_plot.png"), dpi=300)
plt.savefig(os.path.join(OUT, "isomer_plot.svg"))
print(f"\nSaved: {OUT}/isomer_series.csv + isomer_plot.png/.svg")
