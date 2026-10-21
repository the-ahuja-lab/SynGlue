#!/usr/bin/env python3
"""Consolidate 6-PROTAC tool predictions + plot.
Panels: (A) predicted DC50 nM (regressors, log axis), (B) predicted Dmax %, (C) P(active) classifiers.
SVG real text."""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT="/storage/savi/saveenas/Projects/SynGlue/revision/rebuttal_analyses/19_6protacs_toolrun"
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','figure.dpi':300,'savefig.bbox':'tight'})

inp=pd.read_csv(f"{OUT}/input6.csv"); names=inp["name"].tolist(); n=len(names)
# ── load predictions ──
sg_head=pd.read_csv(f"{OUT}/stageA_transformer_heads.csv") if os.path.exists(f"{OUT}/stageA_transformer_heads.csv") else None
# build from known results (already computed):
heads_dc50=np.array([2.273,2.278,1.800,1.676,2.000,1.123])   # log10 nM, frozen transformer heads
heads_dmax=np.array([74.37,72.98,72.36,72.21,82.54,81.38])
rf=pd.read_csv(f"{OUT}/synglue_rf.csv")          # syn glue rf: log dc50 + dmax
mor=pd.read_csv(f"{OUT}/morgan_rf.csv")          # log dc50 + dmax
tnn=pd.read_csv(f"{OUT}/tanimoto_nn.csv")        # log dc50
pdp=pd.read_csv(f"{OUT}/pdp.csv")
pxgb=pd.read_csv(f"{OUT}/pdpxgb.csv")
stan=pd.read_csv(f"{OUT}/protacstan.csv")

rows=[]
for i in range(n):
    rows.append(dict(molecule=names[i],
        SynGlue_head_DC50_nM=float(10**heads_dc50[i]), SynGlue_RF_DC50_nM=float(10**rf["synglue_rf_pred_logdc50"].values[i]),
        MorganRF_DC50_nM=float(10**mor["morgan_rf_pred_logdc50"].values[i]),
        TanimotoNN_DC50_nM=float(10**tnn["tanimoto_nn_pred_logdc50"].values[i]),
        SynGlue_head_Dmax=heads_dmax[i], SynGlue_RF_Dmax=float(rf["synglue_rf_pred_dmax"].values[i]),
        MorganRF_Dmax=float(mor["morgan_rf_pred_dmax"].values[i]),
        PDP_Pactive=float(pdp["pdp_active_proba"].values[i]),
        PDPXGB_Pactive=float(pxgb["pdp_xgb_proba"].values[i]),
        STAN_Pactive=float(stan["protacstan_active_proba"].values[i])))
df=pd.DataFrame(rows)
df.to_csv(f"{OUT}/consolidated_predictions.csv", index=False)
print(df.to_string(index=False))

fig,axes=plt.subplots(1,3,figsize=(14,4.6))
x=np.arange(n); w=0.2
# A: DC50 nM log
ax=axes[0]
d50=[("SynGlue\n(transform.\nhead)",df["SynGlue_head_DC50_nM"].values,"#4C72B0"),
     ("SynGlue\n(frozen RF)",df["SynGlue_RF_DC50_nM"].values,"#5AA9E6"),
     ("Morgan-\nRF",df["MorganRF_DC50_nM"].values,"#55A868"),
     ("Tanimoto-\nNN",df["TanimotoNN_DC50_nM"].values,"#DD8452")]
for k,(lbl,v,c) in enumerate(d50):
    ax.bar(x+(k-1.5)*w, v, width=w*0.9, color=c, label=lbl.replace("\n"," "), log=True, edgecolor='white')
    for xi,vi in zip(x, v): ax.text(xi+(k-1.5)*w, vi*1.12, f"{vi:.0f}", ha='center', fontsize=5.5, rotation=0)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("predicted DC50 (nM, log scale)"); ax.set_title("(A) Predicted DC50 — regressors (lower = more potent)")
ax.legend(fontsize=6, ncol=2, loc='upper left')
# B: Dmax %
ax=axes[1]
dm=[("SynGlue\nhead",df["SynGlue_head_Dmax"].values,"#4C72B0"),
    ("SynGlue\nRF",df["SynGlue_RF_Dmax"].values,"#5AA9E6"),
    ("Morgan-\nRF",df["MorganRF_Dmax"].values,"#55A868")]
for k,(lbl,v,c) in enumerate(dm):
    ax.bar(x+(k-1)*w, v, width=w*0.9, color=c, label=lbl.replace("\n"," "), edgecolor='white')
    for xi,vi in zip(x,v): ax.text(xi+(k-1)*w, vi+1, f"{vi:.0f}", ha='center', fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("predicted Dmax (%)"); ax.set_ylim(0,100); ax.axhline(80, color='#888', ls='--', lw=1)
ax.set_title("(B) Predicted Dmax — regressors")
ax.legend(fontsize=7, loc='lower left')
# C: P(active)
ax=axes[2]
pa=[("PDP",df["PDP_Pactive"].values,"#C44E52"),("PDP-XGB",df["PDPXGB_Pactive"].values,"#E07B6A"),
    ("PROTAC-STATN",df["STAN_Pactive"].values,"#8172B3")]
for k,(lbl,v,c) in enumerate(pa):
    ax.bar(x+(k-1)*w, v, width=w*0.9, color=c, label=lbl, edgecolor='white')
    for xi,vi in zip(x,v): ax.text(xi+(k-1)*w, vi+0.012, f"{vi:.2f}", ha='center', fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("P(active)"); ax.set_ylim(0,1); ax.axhline(0.5, color='#888', ls='--', lw=1)
ax.set_title("(C) Classifiers: P(active)")
ax.legend(fontsize=7, loc='lower left')
fig.suptitle("6 submitted PROTACs — potency predictions across tools (POI: BRD4, E3: VHL)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f"{OUT}/FIG_6protacs_tool_predictions.svg")
fig.savefig(f"{OUT}/FIG_6protacs_tool_predictions.png")
print("saved consolidated + figure")