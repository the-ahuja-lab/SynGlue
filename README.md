# Readme

# MetaboGlue: A Generative AI-Powered Polypharmacology Toolkit

**MetaboGlue** is a Python package designed to facilitate the decoding and designing of complex multi-targeting metabolites using generative AI techniques. It offers a comprehensive suite of tools for researchers and developers working on polypharmacology and drug discovery, including functionalities for metabolite prediction, multi-target interactions, and data visualization.

 <br>
<div align="center">
<img src="images/Asset_2.png"></div>
<br>

---

## Structure-Guided Workflow

MetaboGlue offers a two-part approach:

1. **Data-Driven**
    - Utilize structural databases to extract terminal fragments, ligands, and target mappings.
    - Map input molecules, annotate types and targets, and generate optimized molecules.
2. **Structure-Guided**
    - Input structures from databases like PDB, AlphaFold, or Rosetta.
    - Use the **GCoupler** module to synthesize molecules.
    - Employ warhead selection and scoring tools for optimization.

## Data-Driven Modules

### 1. **Browser**

- Browse through compound data and visualize results.
- Backend: A database containing terminal fragments, ligands, and targets.

### 2. **Mapper**

- Map queries to relevant compound data and retrieve matching ligands and fragments.

### 3. **Annotator**

- Annotate compounds with molecular type, target information, and functional groups.

### 4. **Warhead Mapper**

- Map potential warheads for specific targets during drug design workflows.

### 5. **Generator**

- Generate new molecules based on input data.
- Includes:
    - **Optimizer**: Fine-tune generated structures.
    - **Scorer**: Rank generated molecules.

## Structure-Guided Modules

### 1. **PDB Selection**

- Use protein structures from databases like PDB, AlphaFold, or Rosetta.

### 2. **De Novo Molecule Synthesis**

- Synthesize molecules based on druggable cavities using third-party tools like GCoupler, SiteMap, or Pocket2mol.

### 3. **Warhead Mapper**

- Map and rank the top synthesized molecules for specific targets.

### 4. **Generator**

- Generate new molecules based on input data.
- Includes:
    - **Optimizer**: Fine-tune generated structures.
    - **Scorer**: Rank generated molecules.

## Dependencies

- **fastapi** (v0.95.2) — API server
- **uvicorn** (v0.22.0) — ASGI server for FastAPI
- **requests** (v2.31.0) — HTTP requests
- **pandas** (v1.5.3) — Data manipulation
- **rdkit** (v2023.03.1) — Cheminformatics toolkit
- **setuptools** (v67.6.1) — Python packaging
- **biopython** (v1.81) — Bioinformatics tasks
- **matplotlib** (v3.7.1) — Data visualization
- **scikit-learn** (v1.2.2) — Machine learning
- **seaborn** (v0.12.2) — Data visualization
- **scipy** (v1.10.1) — Scientific computing
- **tqdm** (v4.65.0) — Progress bars

## License

MetaboGlue is licensed under the MIT License. See [LICENSE](https://www.notion.so/saveenasolanki/LICENSE) for more details.

## Acknowledgments

- Thanks to the **RDKit team** for their powerful cheminformatics toolkit.
- Gratitude to the contributors and maintainers of **FastAPI**, **Uvicorn**, and other open-source libraries used in this project.

## Summary

MetaboGlue is a comprehensive toolkit that combines **data-driven** and **structure-guided** methodologies to design and analyze polypharmacological metabolites. It enables researchers to:

- Map and annotate molecules with structural and functional insights.
- Optimize molecular generation workflows for drug discovery.
- Integrate easily into existing pipelines via a RESTful API.

With MetaboGlue, decoding multi-targeting molecule  /metabolites has never been easier!
