#!/usr/bin/env python3
"""
Run PROTAC-STAN on external benchmark compounds NOT in training data.

Pipeline:
1. Filter external compounds where POI_UniProt + E3_UniProt have ESM embeddings
2. Compute RDKit molecular properties
3. Build custom.csv in a data dir under revision/
4. Process into PyG graphs (PROTACData)
5. Load saved model + run inference → degradation class
"""
import os, sys, json, shutil, pickle, warnings
warnings.filterwarnings('ignore')

STAN_DIR = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN"
BASE = "/storage/savi/saveenas/Projects/SynGlue"
DATA_DIR = f"{BASE}/revision/benchmarking/stan_data"
OUT = f"{BASE}/revision/benchmarking/tool_results/protac_stan_external_predictions.csv"
sys.path.insert(0, STAN_DIR)

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Load ESM map from PROTAC-STAN demo data
ESM_MAP_PATH = f"{STAN_DIR}/data/demo/esm_s_map.pkl"
with open(ESM_MAP_PATH, 'rb') as f:
    ESM_MAP = pickle.load(f)
print(f"ESM map loaded: {len(ESM_MAP)} proteins")

# ---- 1. Load external data + filter ----
df = pd.read_csv(f"{BASE}/revision/benchmarking/external_not_in_training.csv", low_memory=False)
mask = df['POI_UniProt'].isin(ESM_MAP.keys()) & df['E3_UniProt'].isin(ESM_MAP.keys())
sub = df[mask].copy()
print(f"Eligible rows (POI+E3 in ESM map): {len(sub)} / {len(df)}")
print(f"Unique SMILES: {sub['SMILES'].nunique()}")

# Deduplicate: one row per (SMILES, POI, E3) — prefer DC50
sub = sub.sort_values('Value_Type')  # DC50 before Dmax
sub = sub.drop_duplicates(subset=['SMILES', 'POI_UniProt', 'E3_UniProt'])
print(f"After dedup: {len(sub)} rows")

# ---- 2. Compute RDKit properties ----
def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    exact = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return {
        'Smiles': smiles,
        'Molecular Weight': round(mw, 4),
        'Exact Mass': round(exact, 4),
        'XLogP3': round(logp, 4),
        'Heavy Atom Count': heavy,
        'Ring Count': rings,
        'Hydrogen Bond Acceptor Count': hba,
        'Hydrogen Bond Donor Count': hbd,
        'Rotatable Bond Count': rotb,
        'Topological Polar Surface Area': round(tpsa, 4),
    }

rows = []
failed = 0
for _, r in sub.iterrows():
    props = calc_props(r['SMILES'])
    if props is None:
        failed += 1
        continue
    props['Compound ID'] = r.get('TPD_ID', '') if pd.notna(r.get('TPD_ID', '')) else ''
    props['Uniprot'] = r['POI_UniProt']
    props['Target'] = r['Protein_of_Interest']
    props['E3 ligase'] = r['E3_Ligase']
    props['E3 ligase Uniprot'] = r['E3_UniProt']
    props['PDB'] = ''
    props['DC50 (nM)'] = r['Value'] if r['Value_Type'] == 'DC50' else ''
    props['Dmax (%)'] = r['Value'] if r['Value_Type'] == 'Dmax' else ''
    props['label'] = 0  # dummy, unused in inference
    props['SMILES_source'] = r['SMILES']
    props['Value_Type_source'] = r['Value_Type']
    props['Value_source'] = r['Value']
    rows.append(props)

print(f"Valid rows: {len(rows)} (failed: {failed})")
stan_df = pd.DataFrame(rows)

# Save the inference CSV (with extra source columns)
stan_df.to_csv(f"{DATA_DIR}/external_inference.csv", index=False)

# ---- 3. Build custom.csv for PROTACData (required columns only) ----
required_cols = [
    'Compound ID', 'Uniprot', 'Target', 'E3 ligase', 'PDB', 'Smiles',
    'DC50 (nM)', 'Dmax (%)', 'Molecular Weight', 'Exact Mass', 'XLogP3',
    'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count',
    'Hydrogen Bond Donor Count', 'Rotatable Bond Count',
    'Topological Polar Surface Area', 'label', 'E3 ligase Uniprot'
]
os.makedirs(f"{DATA_DIR}/custom", exist_ok=True)
custom_csv = f"{DATA_DIR}/custom/custom.csv"
stan_df[required_cols].to_csv(custom_csv, index=False)
print(f"Saved: {custom_csv}")

# Copy ESM map into custom dir (PROTACData.process expects it at root of dataset dir)
shutil.copy(ESM_MAP_PATH, f"{DATA_DIR}/custom/esm_s_map.pkl")

# ---- 4. Process into PyG graphs ----
from data import PROTACData
print("\nProcessing PROTAC graphs...")
dataset = PROTACData(root=f"{DATA_DIR}/custom", name='custom')
print(f"  Graph dataset size: {len(dataset)}")

# ---- 5. Load model + run inference ----
from model import PROTAC_STAN
from data_loader import PROTACLoader

import toml
cfg = toml.load(f"{STAN_DIR}/config.toml")
model = PROTAC_STAN(cfg['model'])
state = torch.load(f"{STAN_DIR}/saved_models/protac-stan.pt", map_location='cpu')
model.load_state_dict(state)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Device: {device}")

_, test_loader = PROTACLoader(root=f"{DATA_DIR}/custom", name='custom', batch_size=8, train_ratio=0.0)
print(f"Test loader size: {len(test_loader.dataset)}")

# Inference
from torch_geometric.data import Batch
all_probs = []
with torch.no_grad():
    for data in test_loader:
        protac_data = data['protac'].to(device)
        e3_ligase_data = data['e3_ligase'].to(device)
        poi_data = data['poi'].to(device)
        outputs, _ = model(protac_data, e3_ligase_data, poi_data, mode='eval')
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())

probs = np.concatenate(all_probs, axis=0)
pred_class = probs.argmax(axis=1)
print(f"\nPredictions: {len(pred_class)}")
print(f"Class distribution: {np.bincount(pred_class)}")
print(f"Mean P(active): {probs[:, 1].mean():.4f}")

# Map back to source rows (order preserved through dataset processing)
out = stan_df.copy().reset_index(drop=True)
out['stan_p_active'] = probs[:, 1]
out['stan_p_inactive'] = probs[:, 0]
out['stan_pred_class'] = pred_class
out.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
print(f"Rows: {len(out)}")
