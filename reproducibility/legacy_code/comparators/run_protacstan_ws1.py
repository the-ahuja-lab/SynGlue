#!/usr/bin/env python3
"""
Run PROTAC-STAN on the FULL external set (covered subset, ESM map available).
E3 UniProt mapping corrected: CRBN=Q96SW2, VHL=P40337 (dataset's Q13698 is CACNA1S — wrong).
Outputs: processed graphs + softmax degradation probabilities.
Usage: conda run -n protac-stan python run_protacstan_ws1.py
"""
import os, sys, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN")
os.chdir("/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN")

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from data import PROTACData
from model import PROTAC_STAN
import toml

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
ROOT = "data/ws1_full"
os.makedirs(os.path.join(ROOT, "processed"), exist_ok=True)

E3_MAP = {"CRBN": "Q96SW2", "VHL": "P40337"}
esm = pickle.load(open("data/custom/esm_s_map.pkl", "rb"))

# FULL external set
ext = pd.read_csv("/storage/savi/saveenas/Projects/SynGlue/external_benchmark_data/final_dataset_E3_known.csv")
sc = ext.drop_duplicates("canonical_SMILES").reset_index(drop=True)
sc = sc.rename(columns={"canonical_SMILES": "canonical_smiles", "E3_Ligase": "E3"})
# keep only covered (POI + E3 in ESM map)
sc = sc[sc["POI_UniProt"].notna() & sc["POI_UniProt"].astype(str).str.upper().isin(esm) &
        sc["E3"].astype(str).str.upper().map(E3_MAP).isin(esm)].reset_index(drop=True)
print(f"compounds for PROTAC-STAN (full, covered): {len(sc)}")

# ESM map coverage check (already loaded above)
missing = set()
for _, r in sc.iterrows():
    if str(r["POI_UniProt"]).upper() not in esm:
        missing.add(str(r["POI_UniProt"]))
    if E3_MAP.get(str(r["E3"]).upper()) not in esm:
        missing.add(E3_MAP.get(str(r["E3"]).upper()))
print("missing uniprots:", missing)
assert not missing, "ESM map incomplete"

def props(smi):
    m = Chem.MolFromSmiles(smi)
    return {
        "Molecular Weight": Descriptors.MolWt(m),
        "Exact Mass": Descriptors.ExactMolWt(m),
        "XLogP3": Crippen.MolLogP(m),
        "Heavy Atom Count": m.GetNumHeavyAtoms(),
        "Ring Count": rdMolDescriptors.CalcNumRings(m),
        "Hydrogen Bond Acceptor Count": rdMolDescriptors.CalcNumHBA(m),
        "Hydrogen Bond Donor Count": rdMolDescriptors.CalcNumHBD(m),
        "Rotatable Bond Count": rdMolDescriptors.CalcNumRotatableBonds(m),
        "Topological Polar Surface Area": rdMolDescriptors.CalcTPSA(m),
    }

rows = []
for _, r in sc.iterrows():
    p = props(r["canonical_smiles"])
    rows.append({"Uniprot": str(r["POI_UniProt"]).upper(),
                 "E3 ligase Uniprot": E3_MAP[str(r["E3"]).upper()],
                 "Smiles": r["canonical_smiles"], **p})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "custom.csv"), index=False)
print(f"wrote {len(df)} rows -> {ROOT}/custom.csv")

# copy esm map into this root
import shutil
shutil.copy("data/custom/esm_s_map.pkl", os.path.join(ROOT, "esm_s_map.pkl"))

# process graphs
print("Processing graphs (PROTACData)...")
ds = PROTACData(root=ROOT, name="custom")
print("processed files:", os.listdir(os.path.join(ROOT, "processed", "custom")))

# inference with softmax probabilities
cfg = toml.load("config.toml")
model = PROTAC_STAN(cfg["model"])
sd = torch.load("saved_models/protac-stan.pt", map_location="cpu")
model.load_state_dict(sd)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device); model.eval()

from data_loader import PROTACLoader
_, loader = PROTACLoader(root=ROOT, name="custom", batch_size=8, train_ratio=0.0)

probs, preds = [], []
with torch.no_grad():
    for data in loader:
        out, _ = model(data["protac"].to(device), data["e3_ligase"].to(device),
                       data["poi"].to(device), mode="eval")
        p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())

sc["protacstan_active_proba"] = probs
sc["protacstan_class"] = preds
out_df = sc[["canonical_smiles", "E3", "POI_UniProt", "protacstan_active_proba", "protacstan_class"]]
out_df.to_csv(os.path.join(TIR, "protacstan_fullset_predictions.csv"), index=False)
print(f"\nSaved {TIR}/protacstan_fullset_predictions.csv ({len(out_df)} rows)")
print(out_df[["canonical_smiles", "protacstan_active_proba"]].head(5).to_string())
