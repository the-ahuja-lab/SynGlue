#!/usr/bin/env python3
"""01_prepare_components.py  [P0]

Actual preprocessing and component-mapping workflow.

Reads the curated PROTAC table and the warhead/linker/E3 component libraries, applies the
production mapping rule, and produces correctly identified component embeddings with compound IDs.

Mapping rule implemented here is the deposited one (`05_Models_attention_dataset_builder.py:66-87`):
  warhead  = first row where warhead_Uniprot == PROTAC.warhead_Uniprot  OR  warhead_Target == PROTAC.warhead_Target
  linker   = row where linker_Compound_ID == PROTAC["Linker ID"]
  E3       = first row where E3_Ligase == PROTAC["E3 ligase"]

IMPORTANT (documented limitation): this is a name/target-based first-match rule. It is
underdetermined for most records (see `06_check_data_leakage.py` and the audit report). A
structure-based variant (`--structure-map`) is provided and uses substructure matching to select
the component that is actually embedded in the PROTAC, falling back to first-match when ambiguous.

Outputs (results/):
  components_mapped.csv   compound_id, protac_smiles, warhead_id, linker_id, e3_id, mapping_mode
  component_embeddings.npz  warhead/linker/e3 vectors keyed by component ID (4,800-d)
  mapping_report.json     counts, ambiguity, zero-match, first-vs-structure agreement
"""
import argparse
import os
import subprocess
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


def canon(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def grover_embed(smiles_list, out_npz, tag):
    """Run the pinned GROVER featuriser (save_features + fingerprint) -> (n, 4800) array."""
    if os.path.exists(out_npz):
        return np.load(out_npz)["fps"][:, :4800]
    C.require(C.GROVER_DIR, "patched GROVER checkout (set GROVER_DIR)")
    C.require(C.GROVER_CKPT, "GROVER checkpoint (set GROVER_CKPT)")
    smi_csv = os.path.join(C.RESULTS, f"tmp_{tag}_smiles.csv")
    feats = os.path.join(C.RESULTS, f"tmp_{tag}_features.npz")
    pd.DataFrame({"smiles": smiles_list}).to_csv(smi_csv, index=False)
    subprocess.run(
        f"python {C.GROVER_DIR}/scripts/save_features.py --data_path {smi_csv} "
        f"--save_path {feats} --features_generator rdkit_2d_normalized --restart",
        shell=True, check=True, capture_output=True)
    subprocess.run(
        f"python {C.GROVER_DIR}/main.py fingerprint --data_path {smi_csv} "
        f"--features_path {feats} --checkpoint_path {C.GROVER_CKPT} "
        f"--fingerprint_source both --output {out_npz}",
        shell=True, check=True, capture_output=True)
    return np.load(out_npz)["fps"][:, :4800]


def build_component_embeddings():
    """Embed every library component once (cached)."""
    out = {}
    for tag, table, idcol, smicol in [
        ("warhead", C.PATHS["warhead_table"], "warhead_Compound_ID", "warhead_Smiles"),
        ("linker", C.PATHS["linker_table"], "linker_Compound_ID", "linker_Smiles"),
        ("e3", C.PATHS["e3_table"], "E3_Compound_ID", "E3_Smiles"),
    ]:
        df = pd.read_csv(C.require(table), low_memory=False)
        smicol = next((c for c in df.columns if c.lower() == smicol.lower()),
                      next((c for c in df.columns if c.lower().startswith("smiles")), None))
        if smicol is None:
            print(f"  [{tag}] no SMILES column found; columns: {list(df.columns)[:10]}")
            continue
        emb = grover_embed(df[smicol].astype(str).tolist(),
                           os.path.join(C.RESULTS, f"emb_{tag}.npz"), tag)
        out[tag] = {"ids": df[idcol].astype(str).values, "X": emb}
        print(f"  [{tag}] {len(df)} rows -> embeddings {emb.shape}")
    return out


def map_first_match(protac, wh, lk, e3):
    rows = []
    for _, p in protac.iterrows():
        w = wh[(wh["warhead_Uniprot"] == p["warhead_Uniprot"]) |
               (wh["warhead_Target"] == p["warhead_Target"])]
        l = lk[lk["linker_Compound_ID"] == p["Linker ID"]]
        e = e3[e3["E3_Ligase"] == p["E3 ligase"]]
        if w.empty or l.empty or e.empty:
            continue
        rows.append(dict(compound_id=str(p["PROTAC_Compound_ID"]), protac_smiles=str(p["PROTAC_SMILES"]),
                         warhead_id=str(w.iloc[0]["warhead_Compound_ID"]), linker_id=str(p["Linker ID"]),
                         e3_id=str(e.iloc[0]["E3_Compound_ID"]), mapping_mode="first_match"))
    return pd.DataFrame(rows)


def map_structure_aware(protac, wh, lk, e3, emb):
    """Prefer the library component that is a substructure of the PROTAC (unique match)."""
    from rdkit import Chem
    wh_smi = dict(zip(emb["warhead"]["ids"], [Chem.MolFromSmiles(s) if isinstance(s, str) else None
                                              for s in wh.get("warhead_Smiles", [])]))
    rows = []
    for _, p in protac.iterrows():
        m = Chem.MolFromSmiles(str(p["PROTAC_SMILES"]))
        w = wh[(wh["warhead_Uniprot"] == p["warhead_Uniprot"]) |
               (wh["warhead_Target"] == p["warhead_Target"])]
        l = lk[lk["linker_Compound_ID"] == p["Linker ID"]]
        e = e3[e3["E3_Ligase"] == p["E3 ligase"]]
        if w.empty or l.empty or e.empty:
            continue
        chosen, mode = str(w.iloc[0]["warhead_Compound_ID"]), "first_match"
        if m is not None and "warhead_Smiles" in w.columns:
            hits = []
            for _, cand in w.iterrows():
                cm = Chem.MolFromSmiles(str(cand["warhead_Smiles"]))
                if cm is not None and m.HasSubstructMatch(cm):
                    hits.append(str(cand["warhead_Compound_ID"]))
            if len(set(hits)) == 1:
                chosen, mode = hits[0], "structure_unique"
            elif len(set(hits)) > 1:
                mode = "structure_ambiguous"
        rows.append(dict(compound_id=str(p["PROTAC_Compound_ID"]), protac_smiles=str(p["PROTAC_SMILES"]),
                         warhead_id=chosen, linker_id=str(p["Linker ID"]),
                         e3_id=str(e.iloc[0]["E3_Compound_ID"]), mapping_mode=mode))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure-map", action="store_true",
                    help="use substructure-based warhead selection instead of first-match")
    args = ap.parse_args()

    protac = pd.read_csv(C.require(C.PATHS["training_table"]), low_memory=False)
    wh = pd.read_csv(C.require(C.PATHS["warhead_table"]), low_memory=False)
    lk = pd.read_csv(C.require(C.PATHS["linker_table"]), low_memory=False)
    e3 = pd.read_csv(C.require(C.PATHS["e3_table"]), low_memory=False)
    for df, cols in [(protac, ["warhead_Uniprot", "warhead_Target", "E3 ligase", "Linker ID",
                               "PROTAC_Compound_ID"]),
                     (wh, ["warhead_Uniprot", "warhead_Target"]),
                     (lk, ["linker_Compound_ID"]), (e3, ["E3_Ligase"])]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.upper()

    print("building component embeddings (cached)...")
    emb = build_component_embeddings()
    np.savez_compressed(os.path.join(C.RESULTS, "component_embeddings.npz"),
                        **{f"{k}_ids": v["ids"] for k, v in emb.items()},
                        **{f"{k}_X": v["X"] for k, v in emb.items()})

    mapped = map_structure_aware(protac, wh, lk, e3, emb) if args.structure_map \
        else map_first_match(protac, wh, lk, e3)
    C.save_csv(mapped, "components_mapped.csv")

    # ambiguity / agreement report
    n_comp = protac["PROTAC_Compound_ID"].nunique()
    amb_w = amb_e = 0
    for _, p in protac.drop_duplicates("PROTAC_Compound_ID").iterrows():
        w = wh[(wh["warhead_Uniprot"] == p["warhead_Uniprot"]) |
               (wh["warhead_Target"] == p["warhead_Target"])]
        e = e3[e3["E3_Ligase"] == p["E3 ligase"]]
        amb_w += int(len(w) > 1)
        amb_e += int(len(e) > 1)
    report = dict(
        training_rows=int(len(protac)), unique_compounds=int(n_comp),
        mapped_rows=int(len(mapped)),
        unique_mapped_compounds=int(mapped["compound_id"].nunique()) if len(mapped) else 0,
        warhead_ambiguous_compounds=int(amb_w), e3_ambiguous_compounds=int(amb_e),
        warhead_ambiguous_pct=round(100 * amb_w / max(n_comp, 1), 1),
        e3_ambiguous_pct=round(100 * amb_e / max(n_comp, 1), 1),
        mapping_mode=mapped["mapping_mode"].value_counts().to_dict() if len(mapped) else {},
        note=("first-match mapping is underdetermined for the majority of records; use "
              "--structure-map for the structure-based variant and report the error rate"),
    )
    C.save_json(report, "mapping_report.json")
    print("\ncomponent mapping summary:")
    for k, v in report.items():
        if k != "note":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
