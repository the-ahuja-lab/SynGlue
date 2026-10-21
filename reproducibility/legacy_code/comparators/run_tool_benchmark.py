#!/usr/bin/env python3
"""
SynGlue Tool Benchmark Runner
Runs predictions on the 4 test compounds across all working tools.
"""
import os, sys, json, warnings, subprocess
warnings.filterwarnings('ignore')

# ========== TEST COMPOUNDS ==========
TEST_COMPOUNDS = {
    "cmp1": "COC1=CC=CC=C1C2=C(C(NC3=NN=C(C#CC4CC4)S3)=O)C=NC(N5CCC6(CN(C7=CC=C(C8CCC(NC8=O)=O)C=C7)C6)CC5)=C2",
    "cmp2": "O=C(C1=C(C2=CC=CC=C2OC)C=C(N3CCN(C4=CC=C(C5CCC(NC5=O)=O)C=C4)C(C)(C)C3)N=C1)NC6=NN=C(C#CC7CC7)S6",
    "BRD4_cmp1": "COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)NCCOCCOCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O",
    "BRD4_cmp2": "COc1cc2c(cc1c1c(C)onc1C)cc(c(=O)n2Cc1ccccn1)C(=O)NCCCCCCCCOCC(=O)N[C@@H](C(C)(C)C)C(=O)N1C[C@@H](C[C@H]1C(=O)N[C@H](c1ccc(cc1)c1scnc1C)C)O"
}

# ========== OUTPUT SETUP ==========
RESULTS_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/tool_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

results_log = {}

# ===================================================================
# TOOL 1: PROTAC-Degradation-Predictor (PDP)
# ===================================================================
def run_pdp():
    """Run PDP predictions on 4 compounds."""
    print("\n" + "="*60)
    print("RUNNING PROTAC-DEGRADATION-PREDICTOR (PDP)")
    print("="*60)
    
    pdp_env = "/home/saveenas/miniconda3/envs/pdp"
    pdp_dir = "/storage/savi/saveenas/Projects/SynGlue/validation_tools/PROTAC-Degradation-Predictor"
    
    # The PDP package needs to be installed in the env
    # First check if it's importable
    check_cmd = f"cd {pdp_dir} && {pdp_env}/bin/python -c 'import protac_degradation_predictor; print(\"PDP importable\")' 2>&1"
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)
    print(f"PDP import check: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"PDP import error: {result.stderr[:200]}")
        # Try installing
        install_cmd = f"cd {pdp_dir} && {pdp_env}/bin/pip install -e . 2>&1 | tail -5"
        subprocess.run(install_cmd, shell=True, capture_output=True, text=True, timeout=60)
    
    pdp_results = {}
    for name, smiles in TEST_COMPOUNDS.items():
        # Try prediction with different E3/target combos
        # For BRD4-targeting PROTACs: target = Q92769 (BRD4), E3 = VHL or CRBN
        # cmp1/cmp2 look like they might target AR or other targets
        
        pred_script = f"""
import sys, json
sys.path.insert(0, '{pdp_dir}')
try:
    import protac_degradation_predictor as pdp
    
    # Available options
    e3s = pdp.avail_e3_ligases()
    uniprots = pdp.avail_uniprots()
    cells = pdp.avail_cell_lines()
    
    result = {{
        'e3_ligases': e3s,
        'uniprots': uniprots[:10],
        'cells': cells[:10],
        'status': 'available'
    }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""
        script_file = f"/tmp/pdp_check_{name}.py"
        with open(script_file, 'w') as f:
            f.write(pred_script)
        
        result = subprocess.run(
            [f"{pdp_env}/bin/python", script_file],
            capture_output=True, text=True, timeout=30
        )
        pdp_results[name] = {"raw_output": result.stdout[:500], "raw_error": result.stderr[:200]}
    
    return pdp_results

# ===================================================================
# TOOL 2: PROTAC-STAN
# ===================================================================
def run_protac_stan():
    """Check PROTAC-STAN setup and inference capability."""
    print("\n" + "="*60)
    print("CHECKING PROTAC-STAN")
    print("="*60)
    
    stan_env = "/home/saveenas/miniconda3/envs/protac-stan"
    stan_dir = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/PROTAC-STAN"
    
    # Check if model loads
    check_script = f"""
import torch, sys
sys.path.insert(0, '{stan_dir}')
from model import PROTAC_STAN
import toml

cfg = toml.load('{stan_dir}/config.toml')
model_cfg = cfg['model']
model = PROTAC_STAN(model_cfg)
state_dict = torch.load('{stan_dir}/saved_models/protac-stan.pt', map_location='cpu')
model.load_state_dict(state_dict)
print("PROTAC-STAN model loaded successfully")
print(f"Parameters: {{sum(p.numel() for p in model.parameters())}}")
"""
    script_file = "/tmp/stan_check.py"
    with open(script_file, 'w') as f:
        f.write(check_script)
    
    result = subprocess.run(
        [f"{stan_env}/bin/python", script_file],
        capture_output=True, text=True, timeout=30
    )
    print(f"STAN check: {result.stdout[:200]}")
    if result.returncode != 0:
        print(f"STAN error: {result.stderr[:300]}")
    
    return {"stdout": result.stdout[:500], "stderr": result.stderr[:300]}

# ===================================================================
# TOOL 3: SynGlue Batch Inference (Target Mapping)
# ===================================================================
def run_synglue_target_mapping():
    """Run SynGlue target-mapping inference."""
    print("\n" + "="*60)
    print("RUNNING SYNGLUE TARGET MAPPING")
    print("="*60)
    
    synglue_env = "/home/saveenas/miniconda3/envs/SynGlue"
    syn_dir = "/storage/savi/saveenas/Projects/SynGlue"
    
    # Check what databases exist
    db_dir = f"{syn_dir}/EMBO_Reproducibility"
    for f in os.listdir(db_dir):
        if f.endswith('.pkl'):
            print(f"  Found DB: {f} ({os.path.getsize(os.path.join(db_dir, f))/1e6:.1f} MB)")
    
    # Create input CSV
    import pandas as pd
    rows = []
    for name, smiles in TEST_COMPOUNDS.items():
        rows.append({"Name": name, "SMILES": smiles})
    df = pd.DataFrame(rows)
    input_csv = f"{RESULTS_DIR}/test_compounds_synglue.csv"
    df.to_csv(input_csv, index=False)
    print(f"Saved input CSV: {input_csv}")
    
    # Run batch inference
    batch_script = f"{syn_dir}/EMBO_Reproducibility/synglue_batch_inference.py.py"
    
    result = subprocess.run(
        [f"{synglue_env}/bin/python", batch_script,
         "--csv", input_csv,
         "--smiles_col", "SMILES",
         "--name_col", "Name",
         "--db_dir", db_dir,
         "--out_dir", f"{RESULTS_DIR}/synglue_output"],
        capture_output=True, text=True, timeout=120
    )
    print(f"SynGlue batch stdout: {result.stdout[:500]}")
    if result.returncode != 0:
        print(f"SynGlue batch error: {result.stderr[:300]}")
    
    return {"stdout": result.stdout[:1000], "stderr": result.stderr[:300]}

# ===================================================================
# TOOL 4: DeepPROTACs (check - needs mol2 inputs)
# ===================================================================
def check_deepprotacs():
    """Check DeepPROTACs - requires 3D structure inputs."""
    print("\n" + "="*60)
    print("CHECKING DeepPROTACs")
    print("="*60)
    
    de_dir = "/storage/savi/saveenas/Projects/SynGlue/Tool_Compare/DeepPROTACs"
    
    print(f"DeepPROTACs requires: ligase_ligand.mol2, ligase_pocket.mol2, target_ligand.mol2, target_pocket.mol2, linker.smi")
    print("Input files must be prepared from 3D protein-ligand complex structures.")
    print("Not runnable from SMILES alone - requires structural biology inputs.")
    
    return {"status": "requires_3d_structures"}

# ===================================================================
# TOOL 5: DegradeMaster (check)
# ===================================================================
def check_degrademaster():
    """Check DegradeMaster - needs complex setup."""
    print("\n" + "="*60)
    print("CHECKING DegradeMaster")
    print("="*60)
    print("DegradeMaster requires: protac graphs, ligase_pocket, target_pocket (all .pt files)")
    print("Needs 3D structural data preparation pipeline.")
    return {"status": "requires_3d_structures"}

# ===================================================================
# TOOL 6: DiffPROTACs (generative)
# ===================================================================
def check_diffprotacs():
    """Check DiffPROTACs - generative model."""
    print("\n" + "="*60)
    print("CHECKING DiffPROTACs")
    print("="*60)
    
    dp_dir = "/storage/savi/saveenas/Projects/SynGlue/validation_tools/DiffPROTACs"
    dp_env = "/home/saveenas/miniconda3/envs/DiffPROTACs"
    
    # Check sample.py for generation
    if os.path.exists(f"{dp_dir}/sample.py"):
        print(f"DiffPROTACs sample.py found - generative model for PROTAC linker design")
        
        check_script = f"""
import sys
sys.path.insert(0, '{dp_dir}')
try:
    import torch
    # Check if checkpoint exists
    print("Checking DiffPROTACs...")
    import os
    ckpt_dir = '{dp_dir}/checkpoints'
    if os.path.exists(ckpt_dir):
        files = os.listdir(ckpt_dir)
        print(f"Checkpoints: {{files}}")
    else:
        print("No checkpoints directory")
except Exception as e:
    print(f"Error: {{e}}")
"""
        script_file = "/tmp/dp_check.py"
        with open(script_file, 'w') as f:
            f.write(check_script)
        
        result = subprocess.run(
            [f"{dp_env}/bin/python", script_file],
            capture_output=True, text=True, timeout=30
        )
        print(f"DiffPROTACs check: {result.stdout[:300]}")
    
    return {"status": "generative_model"}

# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    results = {}
    
    # Tool 1: PDP
    try:
        results['PDP'] = run_pdp()
    except Exception as e:
        results['PDP'] = {"error": str(e)}
    
    # Tool 2: PROTAC-STAN
    try:
        results['PROTAC_STAN'] = run_protac_stan()
    except Exception as e:
        results['PROTAC_STAN'] = {"error": str(e)}
    
    # Tool 3: SynGlue
    try:
        results['SynGlue'] = run_synglue_target_mapping()
    except Exception as e:
        results['SynGlue'] = {"error": str(e)}
    
    # Tool 4-6: Check requirements
    results['DeepPROTACs'] = check_deepprotacs()
    results['DegradeMaster'] = check_degrademaster()
    results['DiffPROTACs'] = check_diffprotacs()
    
    # Save results summary
    summary_path = f"{RESULTS_DIR}/tool_benchmark_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {summary_path}")
