"""
Gcoupler Benchmark — Configuration
Central paths, tool locations, and GSPT1 case parameters.
"""

import os

# ============================================================
# GSPT1 Benchmark Case Inputs
# ============================================================

GSPT1_WARHEAD_SMILES = "*Nc1nnc(C#CC2CC2)s1"
GSPT1_E3_LIGAND_SMILES = "*N1CCN(c2ccc(C3CCC(=O)NC3=O)cc2)CC1"
GSPT1_PAIR_STRING = f"{GSPT1_WARHEAD_SMILES}|{GSPT1_E3_LIGAND_SMILES}"

GSPT1_TARGET_NAME = "GSPT1"
GSPT1_UNIPROT = "P15170"
GSPT1_PDB_PATH = "/storage/savi/saveenas/Projects/Magnet/Gcoupler/GSPT1/5HXB_chainA_clean.pdb"
GSPT1_FULL_PDB = "/storage/savi/saveenas/Projects/Magnet/Gcoupler/GSPT1/5HXB.pdb"

# Known GSPT1 degrader SMILES for novelty/recovery checks
# CC-885 analog, known GSPT1-degrading PROTACs from literature
KNOWN_GSPT1_DEGRADERS = [
    "CC1=C(C(=O)NC2=CC=CC=C2C(=O)N3CCCC3)SC=N1",          # e.g — CC-885 warhead
    "CC1=CC=C(C=C1)C(=O)NCC2=CC(=CC=C2)C3=CC=NC=C3",     # placeholder — known GSPT1 degrader
]

# ============================================================
# SynGlue / Gcoupler Paths
# ============================================================

SYNGLUE_PY_DIR = "/storage/savi/saveenas/Projects/SynGlue_Py"
SYNGLUE_MODELS_DIR = os.path.join(SYNGLUE_PY_DIR, "models")
SYNGLUE_DATA_DIR = os.path.join(SYNGLUE_PY_DIR, "data")
SYNGLUE_REPO = "/storage/savi/saveenas/Projects/SynGlue/repo_source"

GCOUPLER_RESULTS_DIR = "/storage/savi/saveenas/Projects/Magnet/Reinvent/GSPT1/results"
GCOUPLER_CONFIG = "/storage/savi/saveenas/Projects/Magnet/Reinvent/GSPT1/fi_Link-INVENT_Configuration12.json"

# ============================================================
# Validation Tools Paths
# ============================================================

TOOLS_BASE = "/storage/savi/saveenas/Projects/validation_tools"

TOOL_PATHS = {
    "Pocket2Mol": os.path.join(TOOLS_BASE, "Pocket2Mol"),
    "DiffSBDD": os.path.join(TOOLS_BASE, "DiffSBDD"),
    "DiffLinker": os.path.join(TOOLS_BASE, "DiffLinker"),
    "DeLinker": os.path.join(TOOLS_BASE, "DeLinker"),
    "DeepLigBuilder": os.path.join(TOOLS_BASE, "DeepLigBuilder"),
    "PRosettaC": os.path.join(TOOLS_BASE, "PRosettaC"),
    "ShapeLinker": os.path.join(TOOLS_BASE, "ShapeLinker"),
    "PGMG": os.path.join(TOOLS_BASE, "PGMG"),
    "DiffPROTACs": os.path.join(TOOLS_BASE, "DiffPROTACs"),
    "PROTAC-Degradation-Predictor": os.path.join(TOOLS_BASE, "PROTAC-Degradation-Predictor"),
}

# ============================================================
# Conda Environments
# ============================================================

CONDA_BASE = "/home/saveenas/miniconda3/envs"

ENV_PATHS = {
    "synglu_test_package": os.path.join(CONDA_BASE, "synglu_test_package"),
    "reinvent": os.path.join(CONDA_BASE, "reinvent.v3.2"),
    "admet": os.path.join(CONDA_BASE, "admet"),
    "protac_rl": os.path.join(CONDA_BASE, "PROTAC-RL"),
    "protacable": os.path.join(CONDA_BASE, "PROTACable"),
    "env-protac-degradation-predictor": os.path.join(CONDA_BASE, "env-protac-degradation-predictor"),
    "prossettac": os.path.join(CONDA_BASE, "prossettac"),
}

# ============================================================
# Experiment Output
# ============================================================

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Per-tool output dirs
TOOL_RESULTS = {tool: os.path.join(RESULTS_DIR, tool.lower().replace("-", "_"))
                for tool in TOOL_PATHS}
TOOL_RESULTS["Gcoupler"] = os.path.join(RESULTS_DIR, "gcoupler")
TOOL_RESULTS["Pharmacophore"] = os.path.join(RESULTS_DIR, "pharmacophore")

for d in TOOL_RESULTS.values():
    os.makedirs(d, exist_ok=True)

SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ============================================================
# Filter Parameters
# ============================================================

FILTER_PARAMS = {
    "max_mw": 1000,
    "max_logp": 5.0,
    "min_tpsa": 40,
    "max_tpsa": 200,
    "max_rotatable": 15,
    "max_sa_score": 4.0,
    "min_linker_graph_length": 4,
    "max_linker_graph_length": 12,
    "min_linker_effective_length": 4,
    "max_linker_effective_length": 10,
    "max_dc50_nm": 100,
    "min_dmax_pct": 80,
    "n_ro3_violations": 2,
}
