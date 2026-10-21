#!/usr/bin/env python3
"""
Generate the Gcoupler benchmark comparison table + figures.

Aggregates results from all tools, produces:
1. Summary comparison table (Markdown + CSV)
2. Pass-rate waterfall chart
3. Physicochemical property distribution comparison
4. Linker property distribution comparison

Usage:
    conda run -n synglu_test_package python comparison_table.py

Author: Feynman (SynGlue revision team)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, SUMMARY_DIR, TOOL_RESULTS, TOOL_PATHS

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# ============================================================
# Collect results from each tool's filtered output
# ============================================================

def collect_all_results():
    """Load filtered_results.csv from each tool directory."""
    all_summaries = []
    all_filtered = {}

    for tool_name, tool_dir in TOOL_RESULTS.items():
        csv_path = os.path.join(tool_dir, "filtered_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_filtered[tool_name] = df
            total = len(df)
            valid = df["is_valid"].sum() if "is_valid" in df else 0
            passed = df["pass_all"].sum() if "pass_all" in df else 0
            summary = {
                "Tool": tool_name,
                "Total generated": total,
                "Valid SMILES": int(valid),
                "ValidityRate": round(100 * valid / total, 1) if total > 0 else 0.0,
                "PassedAll": int(passed),
                "PassRate": round(100 * passed / total, 1) if total > 0 else 0.0,
            }
            # Additional pass rates
            for col in ["pass_physchem_mw", "pass_physchem_logp", "pass_physchem_tpsa",
                        "pass_sa", "pass_linker_graph", "pass_linker_effective", "pass_novelty"]:
                if col in df.columns:
                    n_pass = df[col].sum()
                    summary[col.replace("pass_", "PassRate_")] = round(100 * n_pass / total, 1) if total > 0 else 0.0
            all_summaries.append(summary)
        else:
            all_summaries.append({
                "Tool": tool_name,
                "Total generated": 0,
                "Valid SMILES": 0,
                "ValidityRate": 0.0,
                "PassedAll": 0,
                "PassRate": 0.0,
                "Status": "Not run / No results",
            })

    return pd.DataFrame(all_summaries), all_filtered


# ============================================================
# Build Comparison Table (Markdown)
# ============================================================

def build_markdown_table(summary_df):
    """Convert summary DataFrame to formatted Markdown."""
    md_lines = [
        "# Benchmark Comparison: Gcoupler vs. Alternative Methods on GSPT1\n",
        "",
        "| Tool | Generated | Valid | Validity% | Passed All | PassRate% | Blocked? |",
        "|------|-----------|-------|-----------|------------|-----------|----------|",
    ]

    has_extended = "PassRate_physchem_mw" in summary_df.columns

    if has_extended:
        md_lines.insert(1, "## Extended Filter Pass Rates\n")
        md_lines.insert(2, "")
        ext_headers = [
            "Tool", "Generated", "Valid", "Validity%", "PassedAll", "PassRate%",
            "MW Pass%", "logP Pass%", "TPSA Pass%", "SA Pass%",
            "LinkerLen Pass%", "Novelty%"
        ]
        md_lines.append("| " + " | ".join(ext_headers) + " |")
        md_lines.append("|" + "|".join(["---"] * len(ext_headers)) + "|")
    else:
        # Simple table
        for _, row in summary_df.iterrows():
            tool = row.get("Tool", "?")
            gen = row.get("Total generated", 0)
            valid = row.get("Valid SMILES", 0)
            vrate = row.get("ValidityRate", 0.0)
            passed = row.get("PassedAll", 0)
            prate = row.get("PassRate", 0.0)
            status = row.get("Status", "")
            blocked = "⚠️" if "Not run" in str(status) or gen == 0 else ""
            md_lines.append(f"| {tool} | {gen} | {valid} | {vrate} | {passed} | {prate} | {blocked} |")

    if has_extended:
        for _, row in summary_df.iterrows():
            tool = row.get("Tool", "?")
            vals = [
                str(row["Total generated"]),
                str(row["Valid SMILES"]),
                str(row["ValidityRate"]),
                str(row["PassedAll"]),
                str(row["PassRate"]),
                str(row.get("PassRate_physchem_mw", "N/A")),
                str(row.get("PassRate_physchem_logp", "N/A")),
                str(row.get("PassRate_physchem_tpsa", "N/A")),
                str(row.get("PassRate_sa", "N/A")),
                str(row.get("PassRate_linker_graph", "N/A")),
                str(row.get("PassRate_novelty", "N/A")),
            ]
            md_lines.append(f"| {tool} | " + " | ".join(vals) + " |")

    return "\n".join(md_lines)


# ============================================================
# Figures
# ============================================================

def plot_pass_rate_waterfall(summary_df, save_path):
    """Plot pass rate waterfall chart across tools."""
    if summary_df.empty or "ValidityRate" not in summary_df.columns:
        print("No data for waterfall plot")
        return

    tools = summary_df["Tool"].tolist()
    validity = summary_df["ValidityRate"].tolist()
    pass_rates = summary_df["PassRate"].tolist()
    generated = summary_df["Total generated"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Validity and pass rates
    x = np.arange(len(tools))
    width = 0.35
    axes[0].bar(x - width/2, validity, width, label="Validity %", color="#4e79a7")
    axes[0].bar(x + width/2, pass_rates, width, label="Pass Rate %", color="#f28e2b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tools, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Filter Pass Rates by Tool")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)

    # Panel 2: Raw counts
    axes[1].bar(x, generated, color="#86bcb6", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tools, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("Number of Candidates")
    axes[1].set_title("Total Generated Candidates")
    axes[1].ticklabel_format(axis='y', style='plain')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall chart: {save_path}")


def plot_property_distributions(all_filtered, save_dir):
    """Plot physicochemical property distributions across tools."""
    valid_tools = {k: v for k, v in all_filtered.items()
                   if v is not None and len(v) > 0 and "mw" in v.columns}

    if len(valid_tools) < 2:
        print("Not enough data for property distribution plots")
        return

    # Combine into single DataFrame
    dfs = []
    for tool, df in valid_tools.items():
        d = df[["mw", "logp", "tpsa", "sa_score", "pass_all"]].copy()
        d["Tool"] = tool
        dfs.append(d)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["mw", "logp", "tpsa"])

    properties = [
        ("mw", "Molecular Weight (Da)", 200, 1000),
        ("logp", "logP", -2, 7),
        ("tpsa", "TPSA (Å²)", 0, 300),
    ]

    fig, axes = plt.subplots(1, len(properties), figsize=(5 * len(properties), 4))

    for ax, (col, label, lo, hi) in zip(axes, properties):
        for tool in combined["Tool"].unique():
            subset = combined[combined["Tool"] == tool]
            sns.kdeplot(subset[col], label=tool, ax=ax, fill=True, alpha=0.3)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "property_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved property distributions: {save_path}")


def plot_linker_distributions(all_filtered, save_dir):
    """Plot linker property distributions."""
    valid_tools = {k: v for k, v in all_filtered.items()
                   if v is not None and len(v) > 0
                   and "graph_length" in v.columns
                   and v["graph_length"].notna().sum() > 0}

    if len(valid_tools) < 2:
        print("Not enough data for linker distribution plots")
        return

    dfs = []
    for tool, df in valid_tools.items():
        d = df[["graph_length", "effective_length"]].copy()
        d["Tool"] = tool
        d = d.dropna()
        dfs.append(d)

    combined = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i, (col, label) in enumerate([
        ("graph_length", "Linker Graph Length (atoms)"),
        ("effective_length", "Linker Effective Length")
    ]):
        for tool in combined["Tool"].unique():
            subset = combined[combined["Tool"] == tool]
            if len(subset) > 0:
                sns.histplot(subset[col], label=tool, ax=axes[i], alpha=0.4, bins=15)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "linker_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved linker distributions: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Gcoupler Benchmark — Comparison Table Builder")
    print("=" * 60)

    summary_df, all_filtered = collect_all_results()
    print(f"\nCollected data for {len(summary_df)} tools")

    # Save CSV
    csv_path = os.path.join(SUMMARY_DIR, "benchmark_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save Markdown
    md = build_markdown_table(summary_df)
    md_path = os.path.join(SUMMARY_DIR, "benchmark_comparison.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved Markdown: {md_path}")
    print("\n" + md)
    print()

    # Generate figures
    plot_pass_rate_waterfall(summary_df, os.path.join(SUMMARY_DIR, "pass_rate_waterfall.png"))
    plot_property_distributions(all_filtered, SUMMARY_DIR)
    plot_linker_distributions(all_filtered, SUMMARY_DIR)
    print(f"\nAll outputs saved to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
