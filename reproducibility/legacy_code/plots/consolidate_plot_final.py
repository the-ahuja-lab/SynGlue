#!/usr/bin/env python3
"""FINAL consolidated figure for 6 PROTACs x all runnable tools.
Panels: A DC50 regressors (log) | B Dmax | C P(active) classifiers. Real SVG text."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os
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
names=pd.read_csv(f"{OUT}/input6.csv")["name"].tolist(); n=len(names)
H50=np.array([2.273,2.278,1.800,1.676,2.000,1.123]); Hdm=np.array([74.37,72.98,72.36,72.21,82.54,81.38])
rf=pd.read_csv(f"{OUT}/synglue_rf.csv"); mor=pd.read_csv(f"{OUT}/morgan_rf.csv"); tnn=pd.read_csv(f"{OUT}/tanimoto_nn.csv")
pdp=pd.read_csv(f"{OUT}/pdp.csv"); pxgb=pd.read_csv(f"{OUT}/pdpxgb.csv"); stan=pd.read_csv(f"{OUT}/protacstan.csv")
tack=pd.read_csv(f"{OUT}/tack.csv"); dc50=pd.read_csv(f"{OUT}/protacdc50.csv")
rows=[]
for i in range(n):
    rows.append(dict(molecule=names[i],
        SynGlue_head_DC50_nM=float(10**H50[i]), SynGlue_RF_DC50_nM=float(10**rf["synglue_rf_pred_logdc50"].values[i]),
        MorganRF_DC50_nM=float(10**mor["morgan_rf_pred_logdc50"].values[i]),
        TanimotoNN_DC50_nM=float(10**tnn["tanimoto_nn_pred_logdc50"].values[i]),
        TACK_XGB_DC50_nM=float(tack["tack_dc50_nM"].values[i]),
        PROTACDC50_DC50_nM=float(dc50["protacdc50_dc50_nM"].values[i]),
        SynGlue_head_Dmax=Hdm[i], SynGlue_RF_Dmax=float(rf["synglue_rf_pred_dmax"].values[i]),
        MorganRF_Dmax=float(mor["morgan_rf_pred_dmax"].values[i]), TACK_XGB_Dmax=float(tack["tack_dmax_pct"].values[i]),
        PDP_Pactive=float(pdp["pdp_active_proba"].values[i]),
        PDPXGB_Pactive=float(pxgb["pdp_xgb_proba"].values[i]),
        STAN_Pactive=float(stan["protacstan_active_proba"].values[i])))
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/consolidated_predictions.csv", index=False)
print(df.round(2).to_string(index=False))

fig,axes=plt.subplots(1,3,figsize=(16,4.8)); x=np.arange(n)
# A: DC50 (all 6 regressor tools, log scale)
reg50=[("SynGlue\nhead",df["SynGlue_head_DC50_nM"].values,"#4C72B0"),
       ("SynGlue\nfrozen RF",df["SynGlue_RF_DC50_nM"].values,"#5AA9E6"),
       ("Morgan-\nRF",df["MorganRF_DC50_nM"].values,"#55A868"),
       ("Tanimoto-\nNN",df["TanimotoNN_DC50_nM"].values,"#DD8452"),
       ("TACK-XGB",df["TACK_XGB_DC50_nM"].values,"#C44E52"),
       ("PROTAC-\nDC50",df["PROTACDC50_DC50_nM"].values,"#8172B3")]
ax=axes[0]; w=0.13
for k,(lbl,v,c) in enumerate(reg50):
    ax.bar(x+(k-2.5)*w, v, width=w*0.88, color=c, label=lbl.replace("\n"," "), log=True, edgecolor='white')
    for xi,vi in zip(x,v): ax.text(xi+(k-2.5)*w, vi*1.15, f"{vi:.1f}", ha='center', fontsize=5)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("predicted DC50 (nM, log)"); ax.set_title("(A) Waterfall: 6 tools predict DC50 (lower = more potent)")
ax.legend(fontsize=6, ncol=2, loc='upper left')
# B: Dmax
regdm=[("SynGlue\nhead",df["SynGlue_head_Dmax"].values,"#4C72B0"),
       ("SynGlue\nRF",df["SynGlue_RF_Dmax"].values,"#5AA9E6"),
       ("Morgan-\nRF",df["MorganRF_Dmax"].values,"#55A868"),
       ("TACK-XGB",df["TACK_XGB_Dmax"].values,"#C44E52")]
ax=axes[1]; w=0.2
for k,(lbl,v,c) in enumerate(regdm):
    ax.bar(x+(k-1.5)*w, v, width=w*0.88, color=c, label=lbl.replace("\n"," "), edgecolor='white')
    for xi,vi in zip(x,v): ax.text(xi+(k-1.5)*w, vi+1, f"{vi:.0f}", ha='center', fontsize=6)
ax.axhline(80,color='#888',ls='--',lw=1); ax.set_ylim(0,100)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("predicted Dmax (%)"); ax.set_title("(B) Dmax — regressors")
ax.legend(fontsize=7, loc='lower left')
# C: classifiers
classi=[("PDP",df["PDP_Pactive"].values,"#C44E52"),("PDP-XGB",df["PDPXGB_Pactive"].values,"#E07B6A"),
        ("PROTAC-STAN",df["STAN_Pactive"].values,"#8172B3")]
ax=axes[2]; w=0.25
for k,(lbl,v,c) in enumerate(classi):
    ax.bar(x+(k-1)*w, v, width=w*0.88, color=c, label=lbl, edgecolor='white')
    for xi,vi in zip(x,v): ax.text(xi+(k-1)*w, vi+0.012, f"{vi:.2f}", ha='center', fontsize=6)
ax.axhline(0.5,color='#888',ls='--',lw=1); ax.set_ylim(0,1.02)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)
ax.set_ylabel("P(active)"); ax.set_title("(C) Classifiers: all six predicted ACTIVE")
ax.legend(fontsize=7, loc='lower left')
fig.suptitle("6 submitted PROTACs (BRD4×VHL) — consensus potency across 8 tools", fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f"{OUT}/FIG_6protacs_tool_predictions.svg")
fig.savefig(f"{OUT}/FIG_6protacs_tool_predictions.png")
print("saved:", f"{OUT}/FIG_6protacs_tool_predictions.svg")