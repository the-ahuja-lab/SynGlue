#!/usr/bin/env python3
"""
PROTAC-DC50 (Petrou & Minhas, Warwick; bioRxiv 2024.09.16.613208) on the external
decomposed subset. Same output units as SynGlue (DC50 nM via log DC50).
Structures: AlphaFold DB per POI UniProt; CRBN/VHL per E3.
Usage: conda run -n protacdc50 bash -c "export LD_LIBRARY_PATH=...; python -s ws2_protacdc50.py"
"""
import os, sys, warnings, pickle, json, urllib.request
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
from torchdrug import data, layers, models
from torchdrug.layers import geometry
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
CACHE = "/tmp/protacdc50/cache"
os.makedirs(CACHE, exist_ok=True)

def get_avalon(smiles):
    calc = FPCalculator("avalon")
    tr = MoleculeTransformer(calc, n_jobs=0)
    return np.array(tr(smiles))

with open("/tmp/protacdc50/gearnet_dc50_predictor.pkl", "rb") as f:
    xgb_model = pickle.load(f)

gcm = layers.GraphConstruction(
    node_layers=[geometry.AlphaCarbonNode()],
    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                 geometry.KNNEdge(k=10, min_distance=5),
                 geometry.SequentialEdge(max_distance=2)],
    edge_feature="gearnet")
gearnet = models.GearNet(input_dim=21, hidden_dims=[512]*6, num_relation=7,
                         edge_input_dim=59, batch_norm=True, activation="relu",
                         concat_hidden=True, short_cut=True, readout="sum", num_angle_bin=8)
gearnet.load_state_dict(torch.load("/tmp/protacdc50/distance_gearnet_edge.pth", map_location="cpu"))
gearnet.eval()

def protein_embedding(uniprot):
    pkl = os.path.join(CACHE, f"emb_{uniprot}.npy")
    if os.path.exists(pkl):
        return np.load(pkl)
    pdb = os.path.join(CACHE, f"{uniprot}.pdb")
    if not os.path.exists(pdb):
        try:
            with urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}", timeout=30) as r:
                info = json.load(r)
            url = info[0]["pdbUrl"]
        except Exception:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v6.pdb"
        urllib.request.urlretrieve(url, pdb)
        if os.path.getsize(pdb) < 1000:
            raise RuntimeError(f"AF download failed for {uniprot}")
    prot = data.Protein.from_pdb(pdb, atom_feature="position", bond_feature="length",
                                 residue_feature="symbol")
    pack = data.Protein.pack([prot])
    with torch.no_grad():
        t = gcm(pack)
        emb = gearnet.forward(t, t.residue_feature.float())["graph_feature"]
    np.save(pkl, emb.numpy())
    return emb.numpy()

E3_MAP = {"CRBN": "Q96SW2", "VHL": "P40337"}

# ─── external decomposed subset ──────────────────────
comp = pd.read_csv(f"{REV}/benchmarking/ws1/ws1_components_v2.csv")
tiers = pd.read_csv(f"{REV}/benchmarking/ws1/ws1_ood_tiers.csv")
ext = pd.read_csv("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv")
ext = ext[~ext["Database"].astype(str).str.contains("PROTAC-DB")]
ep = ext.groupby("canonical_SMILES").apply(lambda g: pd.Series({
    "dc50_median": np.nanmedian(pd.to_numeric(g.loc[g["Value_Type"] == "DC50", "Value"], errors="coerce")) if (g["Value_Type"] == "DC50").any() else np.nan,
    "dmax_max": np.nanmax(pd.to_numeric(g.loc[g["Value_Type"] == "Dmax", "Value"], errors="coerce")) if (g["Value_Type"] == "Dmax").any() else np.nan,
    "dc50_censored": any(g.loc[g["Value_Type"] == "DC50", "Value_Operator"].astype(str).isin([">", "<", ">=", "<="]),),
}))
m = comp.merge(tiers[["canonical_smiles", "tier"]], on="canonical_smiles", how="left")
m = m.merge(ep.reset_index().rename(columns={"canonical_SMILES": "canonical_smiles"}), on="canonical_smiles", how="left")
m = m[m["linker_single_fragment"].fillna(False) & (m["tier"] != "exact_in_train")].reset_index(drop=True)
m["e3_uniprot"] = m["E3_Ligase"].str.upper().map(E3_MAP)
print(f"decomposed clean subset: {len(m)} compounds | POIs: {m['POI_UniProt'].nunique()} | E3s: {m['E3_Ligase'].nunique()}")

# cache protein embeddings
pois = sorted(m["POI_UniProt"].dropna().astype(str).str.upper().unique())
e3s = sorted(m["e3_uniprot"].dropna().unique())
for p in pois + e3s:
    try:
        protein_embedding(p)
        print(f"  embed {p} ok")
    except Exception as e:
        print(f"  embed {p} FAILED: {e}")

# component fingerprints (cache per unique)
def avalon_cached(smi, tag):
    pkl = os.path.join(CACHE, f"fp_{tag}_{abs(hash(smi))}.npy")
    if os.path.exists(pkl):
        return np.load(pkl)
    fp = get_avalon(smi)
    np.save(pkl, fp)
    return fp

rows = []
for i, r in m.iterrows():
    poi = str(r["POI_UniProt"]).upper()
    e3u = r["e3_uniprot"]
    try:
        pe = protein_embedding(poi)
        ee = protein_embedding(e3u)
        wf = avalon_cached(r["warhead_smiles"], "w")
        lf = avalon_cached(r["linker_smiles"], "l")
        ef = avalon_cached(r["e3_smiles"], "e")
        vec = np.column_stack((pe, ee, wf, ef, lf))
        log50 = xgb_model.predict(vec)[0]
        rows.append(dict(canonical_smiles=r["canonical_smiles"], POI_UniProt=poi, E3=r["E3_Ligase"],
                         dc50_median=r["dc50_median"], dmax_max=r["dmax_max"], dc50_censored=r["dc50_censored"],
                         tier=r["tier"], protacdc50_dc50_nM=float(np.exp(log50))))
    except Exception as e:
        rows.append(dict(canonical_smiles=r["canonical_smiles"], POI_UniProt=poi, E3=r["E3_Ligase"],
                         dc50_median=r["dc50_median"], dmax_max=r["dmax_max"], dc50_censored=r["dc50_censored"],
                         tier=r["tier"], protacdc50_dc50_nM=np.nan))
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(m)}")

out = pd.DataFrame(rows)
out.to_csv(f"{REV}/benchmarking/ws1/tiers/protacdc50_predictions.csv", index=False)
print(f"Saved: {REV}/benchmarking/ws1/tiers/protacdc50_predictions.csv ({len(out)} rows, {out['protacdc50_dc50_nM'].notna().sum()} predicted)")
print(out[["canonical_smiles", "protacdc50_dc50_nM", "dc50_median"]].head(5).to_string())
