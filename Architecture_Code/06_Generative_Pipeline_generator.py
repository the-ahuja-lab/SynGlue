import os
import sys
import json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import joblib
import subprocess
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
from rdkit.Chem import RDConfig
from rdkit import RDLogger
from tqdm.auto import tqdm
from IPython.display import display

# Suppress verbose RDKit warnings
RDLogger.DisableLog('rdApp.*')

# =============================================================================
# 1. PHYSICOCHEMICAL, ADME & SYNTHESIZABILITY CALCULATOR
# =============================================================================
try:
    sys.path.append(os.path.join(RDConfig.RDPaths['RDContribDir'], 'SA_Score'))
    import sascorer
    SA_AVAILABLE = True
except:
    SA_AVAILABLE = False

def get_synthesizability(mol):
    if SA_AVAILABLE:
        try:
            return f"SA Score: {round(sascorer.calculateScore(mol), 2)}"
        except:
            pass
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    return f"Fsp3 (Complexity): {round(fsp3, 2)}"

def calculate_adme_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    
    arom_proportion = arom_rings / heavy_atoms if heavy_atoms > 0 else 0
    esol_logs = 0.16 - (0.63 * logp) - (0.0062 * mw) + (0.066 * rot_bonds) - (0.74 * arom_proportion)
    
    return {
        "MW": round(mw, 2), "logP": round(logp, 2), "TPSA": round(tpsa, 2),
        "Flexibility": rot_bonds, "Solubility_LogS": round(esol_logs, 2),
        "Synthesizability": get_synthesizability(mol)
    }

# =============================================================================
# 2. LINKER EXTRACTION & CLASSIFICATION ENGINE (SURGICAL PRECISION)
# =============================================================================
def remove_dummy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    rwmol = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in rwmol.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_indices, reverse=True):
        rwmol.RemoveAtom(idx)
    return rwmol.GetMol()

def extract_linker_smiles(protac_smiles, wh_smi, e3_smi):
    protac = Chem.MolFromSmiles(protac_smiles)
    wh_mol = remove_dummy_atoms(wh_smi)
    e3_mol = remove_dummy_atoms(e3_smi)
    
    if not protac or not wh_mol or not e3_mol: return None
    
    wh_match = protac.GetSubstructMatch(wh_mol)
    e3_match = protac.GetSubstructMatch(e3_mol)
    
    if not wh_match or not e3_match: return None
    if set(wh_match).intersection(set(e3_match)): return None
        
    atoms_to_delete = list(set(wh_match + e3_match))
    rw_mol = Chem.RWMol(protac)
    
    for idx in sorted(atoms_to_delete, reverse=True):
        rw_mol.RemoveAtom(idx)
        
    try:
        Chem.SanitizeMol(rw_mol)
        linker_smiles = Chem.MolToSmiles(rw_mol)
        if not linker_smiles: return None
        return linker_smiles
    except:
        return None

def run_linker_classification(df_candidates, warhead_smi, e3_smi, output_dir, config_dict):
    print("\n✂️ Extracting pure Linkers from PROTACs...")
    df_candidates['Linker_SMILES'] = df_candidates['SMILES'].apply(
        lambda x: extract_linker_smiles(x, warhead_smi, e3_smi)
    )
    
    def is_valid_smiles(smi):
        if not isinstance(smi, str) or not smi: return False
        try:
            return Chem.MolFromSmiles(smi) is not None
        except:
            return False

    df_candidates['Valid_Linker'] = df_candidates['Linker_SMILES'].apply(is_valid_smiles)
    valid_df = df_candidates[df_candidates['Valid_Linker'] == True].copy()
    valid_linkers = valid_df['Linker_SMILES'].tolist()
    
    if not valid_linkers:
        print("❌ Failed to extract any chemically valid linkers.")
        return df_candidates
        
    print(f"🚀 Processing {len(valid_linkers)} valid linkers for Class Prediction...")
    
    smiles_file = os.path.join(output_dir, "temp_linker_smiles.csv")
    features_file = os.path.join(output_dir, "temp_linker_features.npz")
    fingerprint_file = os.path.join(output_dir, "temp_linker_fps.npz")
    
    pd.DataFrame({'smiles': valid_linkers}).to_csv(smiles_file, index=False)
    
    clean_env = os.environ.copy()
    clean_env['MPLBACKEND'] = 'Agg'
    
    print("⏳ Extracting Linker RDKit 2D features...")
    try:
        subprocess.run(
            f"python {config_dict['grover_dir']}/scripts/save_features.py "
            f"--data_path {smiles_file} --save_path {features_file} "
            f"--features_generator rdkit_2d_normalized --restart", 
            shell=True, check=True, env=clean_env, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print("\n❌ GROVER Feature Extraction Failed! Hidden error log:")
        print(e.stderr.decode()[-2000:])
        return df_candidates
    
    print("⏳ Extracting Linker GROVER embeddings...")
    try:
        subprocess.run(
            f"python {config_dict['grover_dir']}/main.py fingerprint "
            f"--data_path {smiles_file} --features_path {features_file} "
            f"--checkpoint_path {config_dict['grover_checkpoint']} "
            f"--fingerprint_source both --output {fingerprint_file}", 
            shell=True, check=True, env=clean_env, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print("\n❌ GROVER Fingerprint Extraction Failed! Hidden error log:")
        print(e.stderr.decode()[-2000:])
        return df_candidates
    
    try:
        grover_fps = np.load(fingerprint_file, allow_pickle=True)['fps'][:, :4800]
    except Exception as e:
        print(f"❌ Failed to load linker embeddings: {e}")
        return df_candidates

    print("🧠 Predicting Linker Classes with Gradient Boosting...")
    model = joblib.load(config_dict['linker_class_model'])
    X_test = pd.DataFrame(grover_fps, columns=model.feature_names_in_)
    
    y_pred = model.predict(X_test)
    y_proba = np.max(model.predict_proba(X_test), axis=1)
    
    pred_dict = dict(zip(valid_linkers, y_pred))
    prob_dict = dict(zip(valid_linkers, y_proba))
    
    df_candidates['Predicted_Linker_Class'] = df_candidates['Linker_SMILES'].map(pred_dict)
    df_candidates['Linker_Class_Prob'] = df_candidates['Linker_SMILES'].map(prob_dict)
    
    for f in [smiles_file, features_file, fingerprint_file]:
        if os.path.exists(f): os.remove(f)
        
    print("✅ Linker Classification complete!")
    return df_candidates

# =============================================================================
# 3. VISUALIZATION ENGINE 
# =============================================================================
def visualize_exit_vectors(warhead_smi, e3_smi, output_dir):
    print("\n--- Visualizing Selected Anchors & Exit Vectors ---")
    os.makedirs(output_dir, exist_ok=True)
    mols = []
    legends = ["Optimized Warhead\n(* = Exit Vector)", "Tagged E3 Ligase\n(* = Exit Vector)"]
    
    for smi in [warhead_smi, e3_smi]:
        mol = Chem.MolFromSmiles(smi)
        if mol: mols.append(mol)
            
    if mols:
        draw_opts = Draw.MolDrawOptions()
        draw_opts.legendFontSize = 24 
        img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(500, 500), legends=legends, returnPNG=False, drawOptions=draw_opts)
        img.save(os.path.join(output_dir, "Exit_Vectors.png"))
        display(img)

def visualize_top_protacs(df, output_dir, top_n=3):
    print(f"\n🏆 --- Drawing Final Top {top_n} PROTAC Candidates --- 🏆")
    
    df_sorted = df.sort_values(by=['Predicted_DC50_nM', 'Predicted_DMax_%'], ascending=[True, False]).reset_index(drop=True)
    top_df = df_sorted.head(top_n).copy()
    top_df['ADME'] = top_df['SMILES'].apply(calculate_adme_properties)
    
    mols, legends = [], []
    for _, row in top_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol and row['ADME']:
            mols.append(mol)
            adme = row['ADME']
            linker_info = f"\nClass: {row.get('Predicted_Linker_Class', 'Unknown')} (Prob: {row.get('Linker_Class_Prob', 0):.2f})" if 'Predicted_Linker_Class' in row else ""
            
            legends.append(
                f"DC50: {row['Predicted_DC50_nM']:.1f} nM | DMax: {row['Predicted_DMax_%']:.1f}%\n"
                f"MW: {adme['MW']} | logP: {adme['logP']} | TPSA: {adme['TPSA']}\n"
                f"{adme['Synthesizability']}{linker_info}"
            )
            
    if mols:
        draw_opts = Draw.MolDrawOptions()
        draw_opts.legendFontSize = 24 
        img = Draw.MolsToGridImage(mols, molsPerRow=min(top_n, 3), subImgSize=(650, 650), legends=legends, returnPNG=False, drawOptions=draw_opts)
        img.save(os.path.join(output_dir, f"Final_Top_{top_n}_Predicted_PROTACs.png"))
        display(img)

# =============================================================================
# 4. OPEN-ADMET AI PROFILER (ENVIRONMENT SCRUBBER BRIDGE)
# =============================================================================
def run_admet_ai(df, output_dir, config_dict, top_n=20):
    print(f"\n🧪 --- Running ADMET-AI Profiling on Top {top_n} Candidates via Subprocess Bridge --- 🧪")
    
    df_sorted = df.sort_values(by=['Predicted_DC50_nM', 'Predicted_DMax_%'], ascending=[True, False]).reset_index(drop=True)
    top_df = df_sorted.head(top_n).copy()
    
    physchem_records = top_df['SMILES'].apply(calculate_adme_properties).tolist()
    physchem_df = pd.DataFrame(physchem_records)
    top_df = pd.concat([top_df.reset_index(drop=True), physchem_df.reset_index(drop=True)], axis=1)
    
    input_csv = os.path.join(output_dir, "temp_admet_input.csv")
    output_csv = os.path.join(output_dir, f"ADMET_Predictions_Top_{top_n}.csv")
    script_path = os.path.join(output_dir, "run_admet_external.py")
    
    top_df.to_csv(input_csv, index=False)
    
    external_script = f"""import os
import matplotlib
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg')

import pandas as pd
from admet_ai import ADMETModel

print("⏳ Initializing ADMETModel weights in 'admet' environment...")
model = ADMETModel()

df = pd.read_csv('{input_csv}')
smiles_list = df['SMILES'].tolist()

print("🔍 Predicting ADMET Properties...")
preds_df = model.predict(smiles=smiles_list)

df = df.reset_index(drop=True)
preds_df = preds_df.reset_index(drop=True)

final_df = pd.concat([df, preds_df], axis=1)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.to_csv('{output_csv}', index=False)
print("✅ ADMET profiling complete!")
"""
    with open(script_path, "w") as f:
        f.write(external_script)
        
    admet_python = config_dict.get("admet_env_python")
    
    clean_env = os.environ.copy()
    clean_env['MPLBACKEND'] = 'Agg'
    keys_to_scrub = ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV', 'CONDA_PREFIX']
    for key in keys_to_scrub:
        clean_env.pop(key, None)
    
    try:
        subprocess.run(f"{admet_python} {script_path}", shell=True, check=True, env=clean_env)
    except subprocess.CalledProcessError as e:
        print(f"❌ ADMET-AI bridge failed.")
        return None
        
    if os.path.exists(output_csv):
        result_df = pd.read_csv(output_csv)
        
        display_cols = [
            "SMILES", "Predicted_DC50_nM", "Predicted_Linker_Class",
            "MW", "logP", "TPSA", "Flexibility", "Synthesizability"
        ]
        
        key_admet = ["QED", "Caco2_Wang", "HIA_Hou", "BBB_Martins", "CYP3A4_Veith", 
                     "hERG", "AMES", "Clearance_Hepatocyte_AZ", "Half_Life_Obach", "LD50_Zhu"]
                     
        for col in key_admet:
            matching_cols = [c for c in result_df.columns if col in c]
            if matching_cols:
                display_cols.append(matching_cols[0])
                
        display_cols = list(dict.fromkeys(display_cols))
        
        print(f"\n📊 --- Comprehensive PhysChem & ADMET Profiles (Top 5 visible) ---")
        display(result_df[display_cols].head(5).round(3))
        
        if os.path.exists(input_csv): os.remove(input_csv)
        if os.path.exists(script_path): os.remove(script_path)
        return result_df
    return None

# =============================================================================
# 5. LINK-INVENT GENERATIVE ENGINE
# =============================================================================
def run_link_invent(pair_string, config_dict):
    print(f"\n[ Link-INVENT ] Processing pair constraints for: {pair_string}")
    output_dir = os.path.join(config_dict["output_dir"], "latest_run")
    os.makedirs(output_dir, exist_ok=True)

    configuration = {
        "version": 3,
        "model_type": "link_invent",
        "run_type": "reinforcement_learning",
        "logging": {
            "sender": "", "recipient": "local",
            "logging_path": os.path.join(output_dir, "progress.log"),
            "result_folder": os.path.join(output_dir, "results"),
            "job_name": "SynGlue_Constrained_Run", "job_id": "N/A"
        },
        "parameters": {
            "actor": os.path.join(config_dict["reinvent_dir"], "models/linkinvent.prior"),
            "critic": os.path.join(config_dict["reinvent_dir"], "models/linkinvent.prior"),
            "warheads": [pair_string],  
            "n_steps": config_dict["n_steps"],
            "learning_rate": 0.0001,
            "batch_size": config_dict["batch_size"], 
            "randomize_warheads": True,
            "learning_strategy": {"name": "dap", "parameters": {"sigma": 120}},
            "scoring_strategy": {
                "name": "link_invent",
                "diversity_filter": {"bucket_size": 25, "minscore": 0, "minsimilarity": 0, "name": "IdenticalMurckoScaffold"},
                "scoring_function": {
                    "name": "custom_product",
                    "parallel": False,
                    "parameters": [
                        {"weight": 2, "component_type": "linker_graph_length", "name": "Linker Graph Length", "specific_parameters": {"transformation": {"high": 12, "low": 4, "transformation_type": "reverse_sigmoid", "k": 0.5}}},
                        {"weight": 2, "component_type": "linker_effective_length", "name": "Linker Effective Length", "specific_parameters": {"transformation": {"high": 8, "low": 4, "transformation_type": "reverse_sigmoid", "k": 0.5}}},
                        {"weight": 2, "component_type": "num_rotatable_bonds", "name": "Flexibility", "specific_parameters": {"transformation": {"high": 12, "low": 0, "transformation_type": "reverse_sigmoid", "k": 0.5}}},
                        {"weight": 1, "component_type": "linker_num_hbd", "name": "Linker Num HBD", "specific_parameters": {"transformation": {"high": 6, "low": 0, "transformation_type": "reverse_sigmoid", "k": 0.15}}},
                        {"weight": 1, "component_type": "linker_num_rings", "name": "Linker Num Rings", "specific_parameters": {"transformation": {"high": 0, "low": 0, "transformation_type": "step"}}},
                        {"weight": 2, "component_type": "molecular_weight", "name": "Total MW", "specific_parameters": {"transformation": {"high": 1000, "low": 700, "transformation_type": "reverse_sigmoid", "k": 0.01}}},
                        {"weight": 2, "component_type": "tpsa", "name": "Total TPSA", "specific_parameters": {"transformation": {"high": 230, "low": 0, "transformation_type": "reverse_sigmoid", "k": 0.1}}}
                    ]
                }
            }
        }
    }

    config_json_path = os.path.join(output_dir, "LinkINVENT_Configuration.json")
    with open(config_json_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=False)

    command = f"{config_dict['reinvent_env']}/bin/python {config_dict['reinvent_dir']}/input.py {config_json_path}"
    print("⏳ Running Generative Latent Space Model... (This may take a few minutes)")
    
    try:
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        scaffold_memory_path = os.path.join(output_dir, 'results/scaffold_memory.csv')
        
        if os.path.exists(scaffold_memory_path):
            df_res = pd.read_csv(scaffold_memory_path)
            print(f"✅ Success! {len(df_res)} tightly constrained PROTACs generated.")
            return df_res, output_dir
        else:
            print("❌ Link-INVENT Warning: No scaffold memory file found.")
            return None, None
            
    except subprocess.CalledProcessError as e:
        print("\n[ FATAL ERROR ]: Link-INVENT failed to execute.")
        return None, None

# =============================================================================
# 6. PHASE 3 PREDICTION ENGINE (GROVER + PYTORCH)
# =============================================================================
class MultiTaskProtacModel(nn.Module):
    def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        h = self.proj(x)                  
        h = self.transformer(h)           
        attn_scores = self.attn_pool(h)   
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1) 
        return self.head_dc50(fused), self.head_dmax(fused), attn_weights

def run_ai_predictions(df_candidates, output_dir, config_dict):
    print("\n🚀 INITIATING PHASE 3: DC50 & DMax PREDICTION PIPELINE...")
    
    smiles_file = os.path.join(output_dir, "temp_candidates.csv")
    features_file = os.path.join(output_dir, "temp_features.npz")
    fingerprint_file = os.path.join(output_dir, "temp_fingerprints.npz")
    
    df_candidates[['SMILES']].rename(columns={'SMILES': 'smiles'}).to_csv(smiles_file, index=False)
    
    clean_env = os.environ.copy()
    clean_env['MPLBACKEND'] = 'Agg'
    
    print("⏳ Extracting GROVER Features... (Step 1/2)")
    subprocess.run(
        f"python {config_dict['grover_dir']}/scripts/save_features.py --data_path {smiles_file} "
        f"--save_path {features_file} --features_generator rdkit_2d_normalized --restart", 
        shell=True, check=True, env=clean_env, capture_output=True
    )
    
    print("⏳ Running GROVER Deep Learning Model... (Step 2/2)")
    subprocess.run(
        f"python {config_dict['grover_dir']}/main.py fingerprint --data_path {smiles_file} "
        f"--features_path {features_file} --checkpoint_path {config_dict['grover_checkpoint']} "
        f"--fingerprint_source both --output {fingerprint_file}", 
        shell=True, check=True, env=clean_env, capture_output=True
    )
    
    new_fingerprints = np.load(fingerprint_file)['fps'][:, :4800]
    
    warhead_df = pd.read_csv(config_dict['warhead_csv'], low_memory=False)
    e3_df = pd.read_csv(config_dict['e3_csv'], low_memory=False)
    
    w_cols = [c for c in warhead_df.columns if c.startswith("Grover_")]
    e_cols = [c for c in e3_df.columns if c.startswith("Grover_")]
    
    constant_warhead_vector = warhead_df[w_cols].iloc[0].values.astype(np.float32)
    constant_e3_vector = e3_df[e_cols].iloc[0].values.astype(np.float32)
    
    num_cands = new_fingerprints.shape[0]
    X_new = np.zeros((num_cands, 3, 4800), dtype=np.float32)
    
    print("⏳ Building Neural Tensor...")
    for i in tqdm(range(num_cands), desc="Assembling 3D Tensors"):
        X_new[i, 0, :] = constant_warhead_vector
        X_new[i, 1, :] = new_fingerprints[i, :]
        X_new[i, 2, :] = constant_e3_vector
        
    print("🎯 Scoring Candidates with PyTorch and Random Forest...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiTaskProtacModel(input_dim=4800).to(device)
    model.load_state_dict(torch.load(config_dict['pt_model'], map_location=device, weights_only=True))
    model.eval()
    
    rf_dc50 = joblib.load(config_dict['rf_dc50_model'])
    rf_dmax = joblib.load(config_dict['rf_dmax_model'])
    
    X_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
    with torch.no_grad():
        h = model.proj(X_tensor)
        h = model.transformer(h)
        attn_scores = model.attn_pool(h)
        attn_weights = torch.softmax(attn_scores, dim=1)
        Z_new = (h * attn_weights).sum(dim=1).cpu().numpy()

    pred_log_dc50 = rf_dc50.predict(Z_new)
    df_candidates['Predicted_DC50_nM'] = 10 ** pred_log_dc50
    df_candidates['Predicted_DMax_%'] = rf_dmax.predict(Z_new)
    
    for f in [smiles_file, features_file, fingerprint_file]:
        if os.path.exists(f): os.remove(f)
        
    final_csv_path = os.path.join(output_dir, "Final_Predicted_PROTACs.csv")
    df_candidates.to_csv(final_csv_path, index=False)
    return df_candidates

# =============================================================================
# 7. SYNGLUE PRIORITIZER ENGINE
# =============================================================================
class SynGlueSelector:
    def __init__(self, e3_df):
        self.e3_df = e3_df.copy()
        self.archetypes = {
            'A_Workhorse': ['CRBN', 'VHL', 'DCAF1'],
            'B_GreaseSink': ['cIAP1', 'cIAP2', 'XIAP', 'IAP', 'MDM2'],
            'C_Covalent': ['RNF4', 'RNF114', 'KEAP1', 'FEM1B', 'DCAF16', 'DCAF11'],
            'D_Planar': ['AhR', 'DCAF15', 'FBXO22', 'KLHL20', 'UBR box', 'KLHDC2']
        }
        for col in ['Molecular Weight', 'Topological Polar Surface Area', 'IC50 (nM)', 'Kd (nM)']:
            if col in self.e3_df.columns:
                self.e3_df[col] = pd.to_numeric(self.e3_df[col], errors='coerce')

    def _windowed_d(self, val, lower, upper, penalty):
        if val < lower: return math.exp(-penalty * ((lower - val) ** 2))
        elif val > upper: return math.exp(-penalty * ((val - upper) ** 2))
        return 1.0

    def score_warheads(self, fragments_df):
        scored_fragments = []
        for idx, row in tqdm(fragments_df.iterrows(), total=len(fragments_df), desc="Scoring Warheads"):
            mol = Chem.MolFromSmiles(row['fragment'])
            if not mol: continue
            
            adme = calculate_adme_properties(row['fragment'])
            w_mpo = (self._windowed_d(adme['MW'], 140, 300, 0.0001) * self._windowed_d(adme['logP'], 1.0, 3.0, 0.5) * self._windowed_d(adme['TPSA'], 0, 60, 0.001)) ** 0.33
            
            row_data = row.to_dict()
            row_data.update({'W_MPO_Score': w_mpo, 'ADME': adme})
            scored_fragments.append(row_data)
        
        scored_fragments.sort(key=lambda x: x['W_MPO_Score'], reverse=True)
        return scored_fragments

    def score_e3s(self, archetype_key):
        valid_targets = self.archetypes.get(archetype_key, [])
        filtered_e3 = self.e3_df[self.e3_df['Target'].isin(valid_targets)].copy()
        scored_e3s = []
        
        for idx, row in tqdm(filtered_e3.iterrows(), total=len(filtered_e3), desc=f"Scoring E3s ({archetype_key})", leave=False):
            mol = Chem.MolFromSmiles(row['Smiles'])
            if not mol: continue
            
            adme = calculate_adme_properties(row['Smiles'])
            d_mw = self._windowed_d(adme['MW'], 250, 500, 0.001)
            d_tpsa = self._windowed_d(adme['TPSA'], 50, 120, 0.001)
            
            affinity = row.get('Kd (nM)') if pd.notna(row.get('Kd (nM)')) else row.get('IC50 (nM)')
            d_aff = 0.1 if pd.isna(affinity) or affinity <= 0 else (1.0 if affinity <= 100.0 else math.exp(-0.000001 * ((affinity - 100.0) ** 2)))
            
            e3_score = (d_mw * d_tpsa * d_aff) ** (1/3)
            row_data = row.to_dict()
            row_data.update({'D_E3_Score': e3_score, 'ADME': adme})
            scored_e3s.append(row_data)
            
        scored_e3s.sort(key=lambda x: x['D_E3_Score'], reverse=True)
        return scored_e3s

    def generate_e3_exit_vector(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return smiles
        pattern = Chem.MolFromSmarts('[$([NH2]),$([NH1]),$([OH]),$([CX3](=O)[OX2H1]),$([c][F,Cl,Br,I])]')
        matches = [m[0] for m in mol.GetSubstructMatches(pattern)]
        if not matches: return smiles 
        
        dist_matrix = Chem.GetDistanceMatrix(mol)
        centroid = np.argmin(dist_matrix.sum(axis=0))
        best_idx = max(matches, key=lambda idx: dist_matrix[centroid][idx])
        
        rw_mol = Chem.RWMol(mol)
        dummy_idx = rw_mol.AddAtom(Chem.Atom(0)) 
        rw_mol.AddBond(best_idx, dummy_idx, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw_mol)
            return Chem.MolToSmiles(rw_mol)
        except Exception:
            return smiles

    def run_selection(self, target_protein, fragments_df):
        print(f"\n--- Running SynGlue Database Selection for Target: {target_protein} ---")
        scored_fragments = self.score_warheads(fragments_df)
        if not scored_fragments: return {"Error": "Failed to score warheads."}
        
        best_wh = scored_fragments[0]
        archetype = 'D_Planar' if best_wh['ADME']['Flexibility'] < 2 else 'A_Workhorse' 
        potential_e3s = self.score_e3s(archetype)
        best_e3 = potential_e3s[0]
        tagged_e3_smiles = self.generate_e3_exit_vector(best_e3['Smiles'])

        print("\n🔬 --- Selected Warhead ADME Profile ---")
        print(f"SMILES: {best_wh['fragment']}")
        print(f"MW: {best_wh['ADME']['MW']} Da | logP: {best_wh['ADME']['logP']} | TPSA: {best_wh['ADME']['TPSA']} Å²")
        print(f"{best_wh['ADME']['Synthesizability']} | MPO Score: {best_wh['W_MPO_Score']:.2f}")
        
        print("\n🔬 --- Selected E3 Ligase ADME Profile ---")
        print(f"Target: {best_e3['Target']} | Affinity: {best_e3.get('Kd (nM)', best_e3.get('IC50 (nM)', 'N/A'))} nM")
        print(f"MW: {best_e3['ADME']['MW']} Da | logP: {best_e3['ADME']['logP']} | TPSA: {best_e3['ADME']['TPSA']} Å²")
        print(f"{best_e3['ADME']['Synthesizability']} | MPO Score: {best_e3['D_E3_Score']:.2f}")
        
        return {
            "Warhead_SMILES": best_wh['fragment'], 
            "E3_Tagged_SMILES": tagged_e3_smiles,
        }

    def run_custom_warhead(self, target_protein, warhead_smiles):
        print(f"\n--- Running Structure-Guided Pipeline for: {target_protein} ---")
        mol = Chem.MolFromSmiles(warhead_smiles)
        if not mol:
            return {"Error": "Invalid SMILES string provided."}
        if len(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]'))) == 0:
            return {"Error": "The custom Warhead SMILES must contain a dummy atom '*' for the linker."}

        adme = calculate_adme_properties(warhead_smiles)
        w_mpo = (self._windowed_d(adme['MW'], 140, 300, 0.0001) * self._windowed_d(adme['logP'], 1.0, 3.0, 0.5) * self._windowed_d(adme['TPSA'], 0, 60, 0.001)) ** 0.33

        wh_dict = {'fragment': warhead_smiles, 'W_MPO_Score': w_mpo, 'ADME': adme}
        archetype = 'D_Planar' if adme['Flexibility'] < 2 else 'A_Workhorse'
        potential_e3s = self.score_e3s(archetype)
        best_e3 = potential_e3s[0]
        tagged_e3_smiles = self.generate_e3_exit_vector(best_e3['Smiles'])

        print("\n🔬 --- Structure-Guided Warhead ADME Profile ---")
        print(f"SMILES: {wh_dict['fragment']}")
        print(f"MW: {wh_dict['ADME']['MW']} Da | logP: {wh_dict['ADME']['logP']} | TPSA: {wh_dict['ADME']['TPSA']} Å²")
        print(f"{wh_dict['ADME']['Synthesizability']} | MPO Score: {wh_dict['W_MPO_Score']:.2f}")

        print("\n🔬 --- Selected E3 Ligase ADME Profile ---")
        print(f"Target: {best_e3['Target']} | Affinity: {best_e3.get('Kd (nM)', best_e3.get('IC50 (nM)', 'N/A'))} nM")
        print(f"MW: {best_e3['ADME']['MW']} Da | logP: {best_e3['ADME']['logP']} | TPSA: {best_e3['ADME']['TPSA']} Å²")
        print(f"{best_e3['ADME']['Synthesizability']} | MPO Score: {best_e3['D_E3_Score']:.2f}")

        return {
            "Warhead_SMILES": wh_dict['fragment'],
            "E3_Tagged_SMILES": tagged_e3_smiles,
        }

# =============================================================================
# 8. INTERACTIVE MASTER EXECUTION 
# =============================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # GLOBAL CONFIGURATION - SYNGLUE_PY BUNDLE PATHS
    # ---------------------------------------------------------
    CONFIG = {
        "e3_db_path": "e3_ligand.csv",
        "fragments_db_path": "warhead_fragments.pkl",
        "reinvent_dir": "reinvent",
        "output_dir": "outputs",
        "batch_size": 16, 
        "n_steps": 100,
        "grover_dir": "grover",
        "grover_checkpoint": "grover_fixed.pt",
        "pt_model": "multitask_transformer.pt",
        "rf_dc50_model": "rf_dc50.joblib",
        "rf_dmax_model": "rf_dmax.joblib",
        "warhead_csv": "grover_warhead.csv",
        "e3_csv": "grover_e3.csv",
        "admet_env_python": "/home/saveenas/miniconda3/envs/admet/bin/python",
        "linker_class_model": "linker_classifier.pkl"
    }
    
    print("1. Loading databases from disk...")
    E3 = pd.read_csv(CONFIG["e3_db_path"])
    AA = pd.read_pickle(CONFIG["fragments_db_path"])
    selector = SynGlueSelector(E3)
    
    print("\n" + "="*55)
    print(" 🚀 SynGlue -> Link-INVENT -> Predictor -> ADMET Master Pipeline")
    print("="*55)
    
    print(" 1: Data-Driven Approach")
    print(" 2: Structure-Guided Approach")
    print(" 3: Exit/Quit")
    
    while True:
        choice = input("\nSelect Option (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']: break
        print("❌ Invalid input. Please type '1', '2', or '3'.")
        
    if choice == '1':
        # Removed the (e.g., BRD4) portion to make the prompt cleaner!
        TARGET_NAME = input("Enter the Target Protein Name [Default: EGFR]: ").strip().upper()
        if not TARGET_NAME:
            TARGET_NAME = "EGFR"
        
        while True:
            threshold_input = input(f"Enter the percentage threshold for {TARGET_NAME} [Default: 75]: ").strip()
            if not threshold_input:
                THRESHOLD = 75.0
                break
            try:
                THRESHOLD = float(threshold_input)
                break
            except ValueError:
                print("❌ Invalid number. Please enter a valid percentage.")
            
        print(f"\n2. Filtering fragments for {TARGET_NAME} at >= {THRESHOLD}%...")
        subset_AA = AA[(AA['Protein'].str.upper() == TARGET_NAME) & (AA['percentage'] >= THRESHOLD)]
        
        if subset_AA.empty:
            print(f"❌ Error: No fragments found for {TARGET_NAME} above {THRESHOLD}%.")
            payload = {"Error": "Empty DataFrame"}
        else:
            payload = selector.run_selection(TARGET_NAME, subset_AA)
            
    elif choice == '2':
        TARGET_NAME = input("Enter a label for this Structure-Guided Target (e.g., Custom_Kinase) [Default: Custom_Target]: ").strip()
        if not TARGET_NAME:
            TARGET_NAME = "Custom_Target"
            
        CUSTOM_SMILES = input("Enter the Warhead SMILES (Must contain a '*'): ").strip()
        payload = selector.run_custom_warhead(TARGET_NAME, CUSTOM_SMILES)

    elif choice == '3':
        payload = {"Error": "User Quit"}

    # --- MAIN EXECUTION FLOW ---
    if "Error" not in payload:
        wh_smi = payload['Warhead_SMILES']
        e3_smi = payload['E3_Tagged_SMILES']
        
        visualize_exit_vectors(wh_smi, e3_smi, CONFIG["output_dir"])
        
        pair_string = f"{wh_smi}|{e3_smi}"
        generated_df, out_path = run_link_invent(pair_string, CONFIG)
        
        if generated_df is not None:
            predicted_df = run_ai_predictions(generated_df, out_path, CONFIG)
            classified_df = run_linker_classification(predicted_df, wh_smi, e3_smi, out_path, CONFIG)
            visualize_top_protacs(classified_df, out_path, top_n=3)
            admet_df = run_admet_ai(classified_df, out_path, CONFIG, top_n=20)
            
        print("\n🎉 [ Full End-to-End Pipeline Complete ] 🎉")
    else:
        print(f"\nPipeline halted: {payload['Error']}")