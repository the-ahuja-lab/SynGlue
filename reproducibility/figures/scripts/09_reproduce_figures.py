#!/usr/bin/env python3
"""09_reproduce_figures.py  [P1]

Regenerate the retained computational figures from saved predictions/metrics.

Every figure is drawn only from machine-readable outputs produced by steps 05-08 (or from saved
projection files), and is skipped with a message when its input is absent. No wet-lab figures are
produced here.

Figures written to ./figures (SVG with editable text + PNG at 300 dpi):
  fig_external_funnel            external population funnel (2,930 -> 2,562 -> 2,484 / 2,431 / 2,256)
  fig_comparator_auroc           per-comparator AUROC at DC50 <= 100 nM (with n labels)
  fig_split_r2                   DC50/Dmax R2 by split (random / scaffold / target / E3)
  fig_activity_density           PROTAC-DB vs external DC50 (log10 nM) density + box
  fig_linker_roc                 linker classifier out-of-fold ROC curves (grouped evaluation)
  fig_leakage_independence       shared compounds and external overlap per split
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C

# Arial-compatible face (Nimbus Sans) with graceful fallback
for _n, _p in {"NimbusSans-Regular": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
               "NimbusSans-Bold": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
        except Exception:
            pass
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "svg.fonttype": "none",
                     "figure.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
TRAIN_C, EXT_C = "#4C72B0", "#C44E52"


def fig_external_funnel():
    p = os.path.join(C.RESULTS, "external_funnel_counts.csv")
    if not os.path.exists(p):
        print("  [funnel] run 07_evaluate_external.py first"); return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    mx = d["n"].max()
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(d)))
    for i, (_, r) in enumerate(d.iterrows()):
        y = len(d) - 1 - i
        ax.barh(y, r["n"] / mx, height=0.62, color=colors[i], edgecolor="white", left=(1 - r["n"] / mx) / 2)
        ax.text(0.5, y, f"{int(r['n']):,}", ha="center", va="center", color="white",
                fontsize=10, weight="bold")
        ax.text(0.01, y + 0.34, r["stage"], fontsize=8, ha="left", va="bottom", color="#222")
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, len(d) - 0.2); ax.axis("off")
    ax.set_title("External benchmark population funnel", fontsize=11)
    C.save_fig(fig, "fig_external_funnel"); plt.close(fig)


def fig_comparator_auroc():
    p = os.path.join(C.RESULTS, "comparator_metrics.csv")
    if not os.path.exists(p):
        print("  [comparator] run 07_evaluate_external.py first"); return
    d = pd.read_csv(p)
    d = d[(d["threshold_nM"] == 100) & d["AUROC"].notna()]
    if not len(d):
        print("  [comparator] no AUROC at 100 nM"); return
    d = d.sort_values("AUROC")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.barh(np.arange(len(d)), d["AUROC"], color="#4C72B0", alpha=0.9, edgecolor="white")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["AUROC"] + 0.01, i, f"{r['AUROC']:.3f}  (n={int(r['n'])})", va="center", fontsize=7.5)
    ax.axvline(0.5, color="#888", ls="--", lw=1)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["comparator"], fontsize=8)
    ax.set_xlim(0.4, 1.02); ax.set_xlabel("AUROC (DC50 $\\leq$ 100 nM)")
    ax.set_title("External comparator performance (per-comparator denominators)", fontsize=10)
    C.save_fig(fig, "fig_comparator_auroc"); plt.close(fig)


def fig_split_r2():
    p = os.path.join(C.RESULTS, "regression_metrics.csv")
    if not os.path.exists(p):
        print("  [split R2] run 05_evaluate_regression.py first"); return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(d["split"].unique()))
    w = 0.38
    for k, (ep, c) in enumerate([("DC50", "#4C72B0"), ("Dmax", "#DD8452")]):
        sub = d[d["endpoint"] == ep]
        ax.bar(x + (k - 0.5) * w, sub["R2"], width=w, color=c, label=ep, edgecolor="white")
        for xi, (_, r) in zip(x + (k - 0.5) * w, sub.iterrows()):
            ax.text(xi, r["R2"] + 0.01, f"{r['R2']:.2f}", ha="center", fontsize=7)
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(d["split"].unique(), fontsize=8)
    ax.set_ylabel("R$^2$ (held-out)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Held-out performance by split", fontsize=10)
    C.save_fig(fig, "fig_split_r2"); plt.close(fig)


def fig_activity_density():
    fp = C.PATHS["external_predictions"]
    if not os.path.exists(fp) or not os.path.exists(C.PATHS["training_table"]):
        print("  [density] inputs missing"); return
    from scipy.stats import gaussian_kde
    tbl = pd.read_csv(C.PATHS["training_table"], low_memory=False)
    v = pd.to_numeric(tbl["DC50 (nM)"], errors="coerce").dropna(); v = v[v > 0]
    train = np.log10(v.values)
    ext = pd.read_csv(fp, low_memory=False)
    e = pd.to_numeric(ext["dc50_median"], errors="coerce").dropna(); e = e[e > 0]
    ext = np.log10(e.values)
    fig, (axd, axb) = plt.subplots(2, 1, figsize=(5.6, 6.0), height_ratios=[2, 1.3])
    for data, c, lbl in [(train, TRAIN_C, "PROTAC-DB (train)"), (ext, EXT_C, "External")]:
        k = gaussian_kde(data)
        xs = np.linspace(min(data.min(), ext.min()), max(data.max(), ext.max()), 300)
        axd.plot(xs, k(xs), color=c, lw=2, label=f"{lbl} (n={len(data)})")
        axd.fill_between(xs, k(xs), color=c, alpha=0.22)
    axd.set_xlabel("log10(DC50 / nM)"); axd.set_ylabel("density"); axd.legend(fontsize=8, frameon=False)
    axd.set_title("DC50 distribution: training vs external", fontsize=10)
    bp = axb.boxplot([train, ext], labels=["PROTAC-DB\n(train)", "External"], patch_artist=True,
                     medianprops=dict(color="k"), showfliers=False, widths=0.5)
    for patch, c in zip(bp["boxes"], [TRAIN_C, EXT_C]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    rng = np.random.RandomState(C.SEED)
    for i, data in enumerate([train, ext]):
        axb.scatter(i + 1 + rng.uniform(-0.16, 0.16, len(data)), data, s=3, alpha=0.25, color="#222")
    axb.set_ylabel("log10(DC50 / nM)")
    C.save_fig(fig, "fig_activity_density"); plt.close(fig)


def fig_linker_roc():
    p = os.path.join(C.RESULTS, "linker_grouped_oof.csv")
    if not os.path.exists(p):
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "outputs", "linker_grouped_oof.csv")
        p = alt if os.path.exists(alt) else p
    if not os.path.exists(p):
        print("  [linker ROC] run 08_train_linker_classifier.py --mode grouped first"); return
    from sklearn.metrics import roc_curve, auc
    d = pd.read_csv(p)
    y = d["true"].values
    cols = [c for c in d.columns if c.startswith("p_class")]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for i, c in enumerate(cols):
        fpr, tpr, _ = roc_curve((y == i).astype(int), d[c])
        ax.plot(fpr, tpr, lw=1.8, label=f"class {i} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Linker classifier — group-disjoint out-of-fold ROC", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    C.save_fig(fig, "fig_linker_roc"); plt.close(fig)


def fig_leakage_independence():
    p = os.path.join(C.RESULTS, "leakage_report.csv")
    if not os.path.exists(p):
        print("  [leakage] run 06_check_data_leakage.py first"); return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(d)); w = 0.38
    ax.bar(x - w / 2, d["shared_compound_ids_train_test"], width=w, color="#C44E52",
           label="shared compound IDs (train ∩ test)", edgecolor="white")
    ax.bar(x + w / 2, d["test_overlap_external"], width=w, color="#55A868",
           label="test ∩ external benchmark", edgecolor="white")
    for xi, v in zip(x - w / 2, d["shared_compound_ids_train_test"]):
        ax.text(xi, v + max(1, d["shared_compound_ids_train_test"].max() * 0.02), str(int(v)),
                ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(d["split"], fontsize=8)
    ax.set_ylabel("compounds"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Partition independence by split", fontsize=10)
    C.save_fig(fig, "fig_leakage_independence"); plt.close(fig)


def main():
    print("regenerating computational figures ->", C.FIGURES)
    for fn in (fig_external_funnel, fig_comparator_auroc, fig_split_r2, fig_activity_density,
               fig_linker_roc, fig_leakage_independence):
        try:
            fn()
        except Exception as e:
            print(f"  [{fn.__name__}] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
