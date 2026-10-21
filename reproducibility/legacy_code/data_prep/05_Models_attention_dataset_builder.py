import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = "00_Database/processed/"
OUTPUT_DIR = "05_Models/"

FILES = {
    "protac": "protac_features.csv",
    "warhead": "warhead_features.csv",
    "linker": "linker_features.csv",
    "e3": "e3_features.csv"
}

# =========================================================
# LOADERS
# =========================================================

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Empty file: {path}")
    print(f"Loaded {os.path.basename(path)} → {df.shape}")
    return df


def load_all():
    return {
        k: load_csv(os.path.join(DATA_DIR, v))
        for k, v in FILES.items()
    }

# =========================================================
# PREPROCESSING
# =========================================================

def normalize(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df


def prefix_grover(df, prefix):
    return df.rename(
        columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")}
    )

# =========================================================
# MAPPING
# =========================================================

def build_protac_map(protac_df, warhead_df, linker_df, e3_df):

    rows = []

    for _, p in protac_df.iterrows():

        # Warhead
        wh = warhead_df[
            (warhead_df["warhead_Uniprot"] == p["warhead_Uniprot"]) |
            (warhead_df["warhead_Target"] == p["warhead_Target"])
        ]
        if wh.empty:
            continue

        # Linker
        lk = linker_df[linker_df["linker_Compound_ID"] == p["Linker ID"]]
        if lk.empty:
            continue

        # E3
        e3 = e3_df[e3_df["E3_Ligase"] == p["E3 ligase"]]
        if e3.empty:
            continue

        rows.append({
            "PROTAC_Compound_ID": p["PROTAC_Compound_ID"],
            "warhead_Compound_ID": wh.iloc[0]["warhead_Compound_ID"],
            "linker_Compound_ID": p["Linker ID"],
            "E3_Compound_ID": e3.iloc[0]["E3_Compound_ID"]
        })

    return pd.DataFrame(rows)

# =========================================================
# FEATURE MERGING
# =========================================================

def attach_features(protac_map, warhead_df, linker_df, e3_df):

    df = (
        protac_map
        .merge(
            warhead_df[["warhead_Compound_ID"] +
                       [c for c in warhead_df.columns if c.startswith("W_Grover_")]],
            on="warhead_Compound_ID"
        )
        .merge(
            linker_df[["linker_Compound_ID"] +
                      [c for c in linker_df.columns if c.startswith("L_Grover_")]],
            on="linker_Compound_ID"
        )
        .merge(
            e3_df[["E3_Compound_ID"] +
                  [c for c in e3_df.columns if c.startswith("E_Grover_")]],
            on="E3_Compound_ID"
        )
    )

    return df

# =========================================================
# BUILD TENSOR
# =========================================================

def build_tensor(df):

    w_cols = [c for c in df.columns if c.startswith("W_Grover_")]
    l_cols = [c for c in df.columns if c.startswith("L_Grover_")]
    e_cols = [c for c in df.columns if c.startswith("E_Grover_")]

    assert len(w_cols) == len(l_cols) == len(e_cols), "Grover dimension mismatch"

    X = np.stack([
        np.stack([
            row[w_cols].values.astype("float32"),
            row[l_cols].values.astype("float32"),
            row[e_cols].values.astype("float32"),
        ])
        for _, row in df.iterrows()
    ])

    return X, w_cols

# =========================================================
# LABEL PROCESSING
# =========================================================

def process_labels(protac_df):

    label_df = protac_df[
        ["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]
    ].copy()

    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"] = pd.to_numeric(label_df["Dmax (%)"], errors="coerce")

    labels = (
        label_df
        .dropna(subset=["DC50 (nM)"])
        .groupby("PROTAC_Compound_ID")
        .agg({
            "DC50 (nM)": "median",
            "Dmax (%)": "max"
        })
        .reset_index()
    )

    return labels

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline():

    data = load_all()

    protac_df = normalize(data["protac"], ["warhead_Uniprot", "warhead_Target", "E3 ligase"])
    warhead_df = normalize(data["warhead"], ["warhead_Uniprot", "warhead_Target"])
    e3_df = normalize(data["e3"], ["E3_Ligase"])
    linker_df = data["linker"]

    # Prefix features
    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df = prefix_grover(linker_df, "L_")
    e3_df = prefix_grover(e3_df, "E_")

    # Mapping
    protac_map = build_protac_map(protac_df, warhead_df, linker_df, e3_df)
    print("Mapped:", protac_map.shape)

    # Features
    features_df = attach_features(protac_map, warhead_df, linker_df, e3_df)
    print("Features:", features_df.shape)

    # Labels
    labels = process_labels(protac_df)

    # Merge
    final_df = features_df.merge(labels, on="PROTAC_Compound_ID", how="left")
    final_df = final_df.dropna(subset=["DC50 (nM)"])

    # Tensor
    X, feature_cols = build_tensor(final_df)

    y_dc50 = np.log10(final_df["DC50 (nM)"].values.astype(float))
    y_dmax = final_df["Dmax (%)"].values.astype(float)

    print("Final X:", X.shape)
    print("Final y_dc50:", y_dc50.shape)
    print("Final y_dmax:", y_dmax.shape)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_dc50.npy"), y_dc50)
    np.save(os.path.join(OUTPUT_DIR, "y_dmax.npy"), y_dmax)

    final_df.to_csv(os.path.join(OUTPUT_DIR, "final_dataset.csv"), index=False)

    print("\nSaved all outputs.")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run_pipeline()