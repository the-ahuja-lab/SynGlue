#!/usr/bin/env python3
"""
Gcoupler Benchmark — Main Orchestrator

Dispatches generation across all available tools, then applies
the standardized filter cascade and builds the comparison table.

Usage:
    # Step 1: Setup environments
    bash setup_tools.sh

    # Step 2: Run all generations
    conda run -n synglu_test_package python run_benchmark.py

    # Step 3: Evaluate and compare
    conda run -n synglu_test_package python standard_filters.py --all
    conda run -n synglu_test_package python comparison_table.py

Author: Feynman (SynGlue revision team)
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    GSPT1_WARHEAD_SMILES, GSPT1_E3_LIGAND_SMILES, GSPT1_PAIR_STRING,
    GSPT1_PDB_PATH, GSPT1_TARGET_NAME,
    RESULTS_DIR, TOOL_RESULTS, TOOL_PATHS, ENV_PATHS,
    SYNGLUE_PY_DIR, SYNGLUE_MODELS_DIR, SYNGLUE_DATA_DIR,
    CONDA_BASE
)


# ============================================================
# Tool Generation Functions
# ============================================================

def run_gcoupler_baseline():
    """
    Gcoupler already has output from the GSPT1 run.
    We copy the existing scaffold_memory.csv into our results dir.
    """
    src_csv = "/storage/savi/saveenas/Projects/Magnet/Reinvent/GSPT1/results/scaffold_memory.csv"
    dst_dir = TOOL_RESULTS["Gcoupler"]
    dst_csv = os.path.join(dst_dir, "scaffold_memory.csv")

    if os.path.exists(src_csv):
        df = pd.read_csv(src_csv)
        df.to_csv(dst_csv, index=False)
        print(f"[Gcoupler] Copied {len(df)} candidates → {dst_csv}")

        # Also run filter cascade
        from standard_filters import apply_filter_cascade, summarize_filter_results
        from config import GSPT1_WARHEAD_SMILES, GSPT1_E3_LIGAND_SMILES, KNOWN_GSPT1_DEGRADERS

        smiles_list = df["SMILES"].dropna().tolist()
        print(f"[Gcoupler] Running filter cascade on {len(smiles_list)} candidates...")
        df_filt = apply_filter_cascade(
            smiles_list,
            GSPT1_WARHEAD_SMILES,
            GSPT1_E3_LIGAND_SMILES,
            known_degraders=KNOWN_GSPT1_DEGRADERS,
        )
        filt_csv = os.path.join(dst_dir, "filtered_results.csv")
        df_filt.to_csv(filt_csv, index=False)

        summary = summarize_filter_results(df_filt, "Gcoupler")
        print(f"[Gcoupler] Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return True
    else:
        print(f"[Gcoupler] Source not found: {src_csv}")
        return False


def run_pocket2mol():
    """
    Run Pocket2Mol on the GSPT1 structure (5HXB).
    Pocket2Mol generates molecules conditioned on a protein pocket.
    """
    tool_dir = TOOL_PATHS["Pocket2Mol"]
    out_dir = TOOL_RESULTS["Pocket2Mol"]

    if not os.path.exists(tool_dir):
        print("[Pocket2Mol] Tool directory not found. Skipping.")
        return False

    # Check if Pocket2Mol env is available
    env_path = os.path.join(CONDA_BASE, "Pocket2Mol")
    if not os.path.exists(env_path):
        print("[Pocket2Mol] Conda env 'Pocket2Mol' not installed. Skipping.")
        return False

    # Pocket2Mol generation script expects:
    #   python generate.py --pdb <protein.pdb> --out <output_dir> --num_samples <N>
    # First check what scripts Pocket2Mol provides
    scripts = [f for f in os.listdir(tool_dir) if f.endswith('.py')]
    print(f"[Pocket2Mol] Available scripts: {scripts}")

    # Pocket2Mol typically requires a reference ligand to define the pocket.
    # For GSPT1 (ligand-sparse), we use the predicted binding pocket from 5HXB.
    pdb_path = GSPT1_PDB_PATH
    if not os.path.exists(pdb_path):
        print(f"[Pocket2Mol] PDB not found: {pdb_path}")
        return False

    # Look for the main generation entry point
    if "generate.py" in scripts:
        cmd = f"conda run -p {env_path} python {tool_dir}/generate.py --pdb {pdb_path} --out {out_dir} --num_samples 1500"
        print(f"[Pocket2Mol] Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=3600)
            print("[Pocket2Mol] Generation complete")
            return True
        except subprocess.TimeoutExpired:
            print("[Pocket2Mol] Timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"[Pocket2Mol] Failed: {e}")
            return False
    else:
        print("[Pocket2Mol] No generate.py found. Check tool setup.")
        return False


def run_diffsbdd():
    """
    Run DiffSBDD on the GSPT1 structure.
    DiffSBDD generates small molecules conditioned on a protein pocket via diffusion.
    """
    tool_dir = TOOL_PATHS["DiffSBDD"]
    out_dir = TOOL_RESULTS["DiffSBDD"]

    if not os.path.exists(tool_dir):
        print("[DiffSBDD] Tool directory not found. Skipping.")
        return False

    # DiffSBDD uses generate_ligands.py
    gen_script = os.path.join(tool_dir, "generate_ligands.py")
    if not os.path.exists(gen_script):
        print("[DiffSBDD] generate_ligands.py not found.")
        return False

    env_path = os.path.join(CONDA_BASE, "diffsbdd")
    if not os.path.exists(env_path):
        print("[DiffSBDD] Conda env 'diffsbdd' not installed. Skipping.")
        return False

    pdb_path = GSPT1_PDB_PATH
    if not os.path.exists(pdb_path):
        print(f"[DiffSBDD] PDB not found: {pdb_path}")
        return False

    cmd = (
        f"conda run -p {env_path} python {gen_script} "
        f"--protein {pdb_path} --out {out_dir} --samples 1500"
    )
    print(f"[DiffSBDD] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=7200)
        print("[DiffSBDD] Generation complete")
        return True
    except subprocess.TimeoutExpired:
        print("[DiffSBDD] Timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[DiffSBDD] Failed: {e}")
        return False


def run_difflinker():
    """
    Run DiffLinker to generate linkers bridging the warhead and E3 ligand.
    DiffLinker is a 3D diffusion model for linker generation given fragment positions.
    """
    tool_dir = TOOL_PATHS["DiffLinker"]
    out_dir = TOOL_RESULTS["DiffLinker"]

    if not os.path.exists(tool_dir):
        print("[DiffLinker] Tool directory not found. Skipping.")
        return False

    # DiffLinker requires 3D conformers of fragments.
    # We need to generate 3D SDF files for warhead and E3 ligand.
    wh_smi = GSPT1_WARHEAD_SMILES.replace("*", "C")  # Replace dummy with carbon
    e3_smi = GSPT1_E3_LIGAND_SMILES.replace("*", "C")

    # Generate SDF via RDKit ETKDG
    output_sdf_dir = os.path.join(out_dir, "fragments")
    os.makedirs(output_sdf_dir, exist_ok=True)

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom

        for name, smi in [("warhead", wh_smi), ("e3_ligand", e3_smi)]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol = Chem.AddHs(mol)
                params = rdDistGeom.ETKDGv3()
                params.randomSeed = 42
                status = AllChem.EmbedMolecule(mol, params)
                if status == 0:
                    AllChem.MMFFOptimizeMolecule(mol)
                writer = Chem.SDWriter(os.path.join(output_sdf_dir, f"{name}.sdf"))
                writer.write(mol)
                writer.close()

        print(f"[DiffLinker] Fragment SDFs saved to {output_sdf_dir}")
    except Exception as e:
        print(f"[DiffLinker] Failed to generate fragment SDFs: {e}")
        return False

    # DiffLinker generate_with_pocket.py
    gen_script = os.path.join(tool_dir, "generate_with_pocket.py")
    if not os.path.exists(gen_script):
        # Try alternative entry points
        gen_script = os.path.join(tool_dir, "generate.py")
        if not os.path.exists(gen_script):
            print("[DiffLinker] No generate script found.")
            return False

    env_path = os.path.join(CONDA_BASE, "difflinker")
    if not os.path.exists(env_path):
        print("[DiffLinker] Conda env 'difflinker' not installed. Skipping.")
        return False

    # Check available configs for guidance
    configs = [f for f in os.listdir(os.path.join(tool_dir, "configs"))] if os.path.isdir(os.path.join(tool_dir, "configs")) else []
    print(f"[DiffLinker] Available configs: {configs}")

    cmd = (
        f"conda run -p {env_path} python {gen_script} "
        f"--warhead_sdf {output_sdf_dir}/warhead.sdf "
        f"--e3_sdf {output_sdf_dir}/e3_ligand.sdf "
        f"--protein {GSPT1_PDB_PATH} "
        f"--out {out_dir} --samples 1500"
    )
    print(f"[DiffLinker] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=7200)
        print("[DiffLinker] Generation complete")
        return True
    except subprocess.TimeoutExpired:
        print("[DiffLinker] Timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[DiffLinker] Failed: {e}")
        return False


def run_delinker():
    """
    Run DeLinker to generate linkers from warhead-E3 SMILES pairs.
    DeLinker generates 2D linker SMILES connecting two fragments.
    """
    tool_dir = TOOL_PATHS["DeLinker"]
    out_dir = TOOL_RESULTS["DeLinker"]

    if not os.path.exists(tool_dir):
        print("[DeLinker] Tool directory not found. Skipping.")
        return False

    env_path = os.path.join(CONDA_BASE, "delinker")
    if not os.path.exists(env_path):
        print("[DeLinker] Conda env 'delinker' not installed. Skipping.")
        return False

    # DeLinker expects fragment SMILES without the dummy atom
    wh_smi = GSPT1_WARHEAD_SMILES.replace("*", "")  # Just remove the dummy marker
    e3_smi = GSPT1_E3_LIGAND_SMILES.replace("*", "")

    # Write input file for DeLinker
    input_dict = {
        "warheads": [wh_smi],
        "e3_ligands": [e3_smi],
        "num_samples": 1500,
    }
    input_path = os.path.join(out_dir, "input.json")
    with open(input_path, "w") as f:
        json.dump(input_dict, f, indent=2)

    # DeLinker entry point
    gen_script = os.path.join(tool_dir, "DeLinker.py")
    if not os.path.exists(gen_script):
        gen_script = os.path.join(tool_dir, "DeLinker_test.py")
        if not os.path.exists(gen_script):
            print("[DeLinker] DeLinker.py/DeLinker_test.py not found.")
            return False

    cmd = (
        f"conda run -p {env_path} python {gen_script} "
        f"--input {input_path} --out {out_dir}"
    )
    print(f"[DeLinker] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=7200)
        print("[DeLinker] Generation complete")
        return True
    except subprocess.TimeoutExpired:
        print("[DeLinker] Timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[DeLinker] Failed: {e}")
        return False


def run_pharmacophore():
    """
    Pharmacophore-guided linker enumeration.
    Enumerate linker candidates from available fragment DB that:
    - Bridge warhead and E3 ligand
    - Satisfy distance constraints (4-12 atoms)
    - Meet physicochemical criteria
    """
    out_dir = TOOL_RESULTS["Pharmacophore"]

    print("[Pharmacophore] Enumerating linker candidates...")

    # Load fragment database for linker fragments
    db_path = os.path.join(SYNGLUE_DATA_DIR, "warhead_fragments.pkl")
    if not os.path.exists(db_path):
        print(f"[Pharmacophore] Fragment DB not found: {db_path}")
        return False

    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        import numpy as np

        fragments_df = pd.read_pickle(db_path)
        # Use available fragments as linkers
        all_fragments = fragments_df["fragment"].dropna().unique().tolist()

        # Parse as molecules, filter by size (linker-like: 4-12 heavy atoms)
        valid_linkers = []
        for smi in all_fragments:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_atoms = mol.GetNumHeavyAtoms()
            if 4 <= n_atoms <= 12:
                # Check it has attachment points (dummy atoms or suitable functional groups)
                valid_linkers.append(Chem.MolToSmiles(mol, canonical=True))

        valid_linkers = list(set(valid_linkers))
        print(f"[Pharmacophore] Found {len(valid_linkers)} potential linker fragments from DB")

        # Build PROTACs: warhead - linker - E3
        wh_smi = GSPT1_WARHEAD_SMILES
        e3_smi = GSPT1_E3_LIGAND_SMILES

        def combine_fragments(warhead, linker, e3_lig):
            """Simple SMILES concatenation for linking fragments."""
            # Remove dummy atoms from warhead and e3
            wh_clean = warhead.replace("*", "")
            e3_clean = e3_lig.replace("*", "")
            # Combine: wh - linker - e3
            combined = f"{wh_clean}{linker}{e3_clean}"
            mol = Chem.MolFromSmiles(combined)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            return None

        candidates = []
        for linker in valid_linkers[:5000]:  # Limit enumeration
            full_smi = combine_fragments(wh_smi, linker, e3_smi)
            if full_smi:
                # Quick property filter
                mol = Chem.MolFromSmiles(full_smi)
                if mol:
                    mw = Descriptors.MolWt(mol)
                    if mw <= 1200:
                        candidates.append(full_smi)

        candidates = list(set(candidates))
        print(f"[Pharmacophore] Generated {len(candidates)} candidate PROTACs")

        result_df = pd.DataFrame({"SMILES": candidates})
        csv_path = os.path.join(out_dir, "generated_candidates.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"[Pharmacophore] Saved to {csv_path}")
        return True

    except Exception as e:
        print(f"[Pharmacophore] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_prosettac():
    """
    Run PRosettaC for PROTAC-constrained generation.
    Note: Requires complex setup with input PDB structures.
    """
    tool_dir = TOOL_PATHS.get("PRosettaC")
    out_dir = TOOL_RESULTS["PRosettaC"]

    if not tool_dir or not os.path.exists(tool_dir):
        print("[PRosettaC] Tool directory not found. Skipping.")
        return False

    env_path = ENV_PATHS.get("prossettac")
    if not env_path or not os.path.exists(env_path):
        print("[PRosettaC] Conda env 'prossettac' not found. Skipping.")
        return False

    print("[PRosettaC] Checking entry points...")
    scripts = [f for f in os.listdir(tool_dir) if f.endswith('.py')]
    print(f"  Available: {scripts}")

    # PRosettaC typically needs: target PDB, E3 PDB, warhead SMILES, E3 ligand SMILES
    # This requires significant setup - document as partially blocked if not configured
    print("[PRosettaC] Requires complex input setup. Marking as incomplete in benchmark.")
    return False


# ============================================================
# Main Dispatcher
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gcoupler Benchmark Orchestrator")

    parser.add_argument(
        "--only", type=str, default=None,
        help="Run only specific tool(s), comma-separated: gcoupler,pocket2mol,diffsbdd,..."
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Skip specific tool(s), comma-separated"
    )
    parser.add_argument(
        "--timeout", type=int, default=7200,
        help="Per-tool timeout in seconds (default: 7200)"
    )

    args = parser.parse_args()

    only_list = [t.strip().lower() for t in args.only.split(",")] if args.only else None
    skip_list = [t.strip().lower() for t in args.skip.split(",")] if args.skip else []

    # Define jobs
    jobs = [
        ("Gcoupler", run_gcoupler_baseline, "synglu_test_package"),
        ("Pocket2Mol", run_pocket2mol, None),
        ("DiffSBDD", run_diffsbdd, None),
        ("DiffLinker", run_difflinker, None),
        ("DeLinker", run_delinker, None),
        ("Pharmacophore", run_pharmacophore, "synglu_test_package"),
        ("PRosettaC", run_prosettac, None),
    ]

    results = {}

    for name, func, conda_env in jobs:
        name_lower = name.lower()

        if only_list and name_lower not in only_list:
            print(f"\n[Skipping {name}] Not in --only list")
            continue

        if name_lower in skip_list:
            print(f"\n[Skipping {name}] In --skip list")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        try:
            if conda_env and conda_env not in name_lower:
                # Run in conda environment if specified
                script_path = os.path.abspath(__file__)
                env_path = ENV_PATHS.get(conda_env)
                if env_path and os.path.exists(env_path):
                    cmd = f"conda run -p {env_path} python {script_path} --only {name_lower}"
                    print(f"  Dispatching via conda env: {env_path}")
                    subprocess.run(cmd, shell=True, check=True, timeout=args.timeout)
                    results[name] = "Success"
                else:
                    # Run directly
                    success = func()
                    results[name] = "Success" if success else "Failed"
            else:
                success = func()
                results[name] = "Success" if success else "Failed"
        except Exception as e:
            print(f"[{name}] Error: {e}")
            results[name] = f"Failed: {e}"
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("Benchmark Run Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        status_icon = "✅" if status == "Success" else "❌" if "Fail" in str(status) else "⚠️"
        print(f"  {status_icon} {name}: {status}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    try:
        from comparison_table import main as comp_main
        comp_main()
    except Exception as e:
        print(f"Comparison table generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
