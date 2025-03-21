
# SynGlue: A Generative AI Toolkit for PROTAC Design

**SynGlue** is a powerful Python package built to accelerate the discovery and design of complex PROTACs (Proteolysis Targeting Chimeras) using generative AI. Tailored for researchers and developers in polypharmacology and drug discovery, SynGlue offers an end-to-end suite of tools for generating PROTACs and prioritising .

<br>
<div align="center">
<img src="images/Asset_2.png"></div>
<br>


<div align="left">

<p>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9-blue">
  <img src="https://img.shields.io/badge/pypi-v0.1.6-orange">
  <img src="https://img.shields.io/conda/vn/conda-forge/YOUR_PACKAGE">
  <a href="https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/notebook.ipynb">
    <img src="https://img.shields.io/badge/Notebook-Run-black">
  </a>
</p>

</div>




## Features

- **Generative AI Models**: Leverage AI to design and optimize PROTACs and Multitargeting Molecules.
- **ALgorithm** : Fast Fragment Based TRIE data storage algorithm.
- **Polypharmacology**: Classify molecules based on type and target mapping.
- **Flexible API**: Easily integrate SynGlue into your workflows with a RESTful API powered by FastAPI.
- **Cheminformatics Support**: Tools for molecular representation, manipulation, and analysis powered by RDKit.
- **Modular Design**: SynGlue is built with multiple modules to perform tasks like browsing, computing , annotating, mapping  and molecules.
  
# Installation
You can install SynGlue using any of the following methods.

**The package installation takes a few seconds to finish.**

### Install from PyPI

```bash
pip install SynGlue

```

### Clone and Install Locally

```bash
git clone <https://github.com/the-ahuja-lab/SynGlue.git>
cd SynGlue
pip install .

```

### With `conda`:
```bash
conda install -c conda-forge synglue
```

The conda-forge package for synglue is maintained here.



## SynGlue Workflow

SynGlue offers a two-part approach:

- **Data-Driven**
    - Utilize structural databases to extract terminal fragments, ligands, and target mappings.
    - Map input molecules, annotate types and targets, and generate optimized molecules.
- **Structure-Guided**
    - Input structures from databases like PDB, AlphaFold, or Rosetta.
    - Use the **GCoupler** module to synthesize molecules till Authenticator.
    - Employ warhead selection and scoring tools for optimization.

## 1.  Data-Driven Modules

### 1.1 --- **MagnetDB Database**

- Browse through compound data and visualize results.
- Backend: A database containing terminal fragments, ligands, and their targets.

```bash
>>> import MagnetDatabase as db
```

### 1.2 --- **Computator**

- Map queries to relevant compound data and retrieve matching ligands and fragments.

```bash
>>> import Computator as comp
```


### 1.3 --- **Annotator**

- Annotate compounds with molecular type, target information, and functional groups.
```bash
>>> import Annotator as ant
```
To calculate the types and functional groups  of mapped compounds

```bash
>>>ant.types(path='pre-set default Output folder/')
```

To change Annotator strigency default is query fargment>= 25% and Target Fragment 75%
```bash
>>> ant.thershold('query_fragment >=x' ; target_fragment >=y)
```

#### Type Analysis

| Classification                  | Type  |
|----------------------------------|-------|
| **Monovalent, Monotarget**       | Type 1 |
| **Monovalent, Multitarget**      | Type 2 |
| **Multivalent, Monotarget**      | Type 3 |
| **Multivalent, Multitarget**     | Type 4 |


### 1.4 --- **Warhead Mapper and Generator**

- Map potential warheads for specific targets during drug design workflows.
```bash
>>> import Warhead Mapper as wm
```
  
```bash
>>> wm.targets('Target1', E3ligand)
```

- Generate new molecules based on input data.
- Includes:
    - **Optimizer**: Fine-tune generated structures.
    - **Scorer**: Rank generated molecules.
 
```bash
>>> import Generator as gn
```


## 2.  Structure-Guided Modules

### 2.1    **PDB Selection**

- Use protein structures from databases like PDB, AlphaFold, or Rosetta.

### 2.2    **De Novo Molecule Synthesis**

- Synthesize molecules based on druggable cavities using third-party tools like GCoupler, SiteMap, or Pocket2mol.

### 2.3   **Warhead Mapper**

- Map and rank the top synthesized molecules for specific targets.

### 2.4    **Generator**

- Generate new molecules based on input data.
- Includes:
    - **Optimizer**: Fine-tune generated structures.
    - **Scorer**: Rank generated molecules.


Additional arguments:
1. DC50 : Transformer based regressor to predict DC50 values 
2. Dmax : Transformer based regressor to predict Dmax vaues
3. DC50 Classifier : Multiclass prediction of DC50 based on linker

##### Output folder
The output folder will contain the following files at the end of the successful execution of the Generator module
| Files | Description |
| -------- | -------- |
| CSV files | Generated molecules with the predicted DC50 , Dmax |



## Tutorials
To run SynGlue, we have prepared a set of tutorials to help you get started. These tutorials are designed for beginners and eaily runnable.
You can run them directly in Google Colab using the links below. However, you will need an license key to use SynGlue.


| Tutorial | Difficulty | Colab Link |
|----------|------------|------------|
| 1. Magnet Database  | ![Level](https://img.shields.io/badge/Level-Beginner-green) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK) |
| 2. Data Driven Computator and Annotator | ![Level](https://img.shields.io/badge/Level-Beginner-green) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK) |
| 3. Data Driven Generator  | ![Level](https://img.shields.io/badge/Level-Intermediate-yellow) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK) |
| 4. PROTAC Priortization | ![Level](https://img.shields.io/badge/Level-Beginner-green) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK) |
| 5. Structure Guided Generator | ![Level](https://img.shields.io/badge/Level-Intermediate-yellow) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK) |

## Summary

SynGlue is a comprehensive toolkit that combines **data-driven** and **structure-guided** methodologies to design and analyze PROTACs. It enables researchers to:

- Map and annotate molecules with structural and functional insights.
- Optimize molecular generation workflows for drug discovery.
- Integrate easily into existing pipelines via a RESTful API.

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

SynGlue is licensed under the MIT License. See [LICENSE](https://www.notion.so/saveenasolanki/LICENSE) for more details.


## Resources

### Inspiration  
Many existing works greatly inspired this project! Here is a non-exhaustive list:

- 📚 [Reinvent4](https://github.com/MolecularAI/REINVENT4) — A molecular design tool for linker design.  
- 📚 [DeepChem](https://deepchem.readthedocs.io/en/latest/api_reference/models.html) — A pioneer in writing DL programs in many different ways! Has been a huge inspiration for us.  
- 📚 [GCoupler](https://github.com/the-ahuja-lab/Gcoupler) — An integrative approach combining de novo ligand design, statistical methods, and Graph Neural Networks for rational prediction of high-affinity ligands.  


-----------------------

## Acknowledgments

- Thanks to the **RDKit team** for their powerful cheminformatics toolkit.
- Gratitude to the contributors and maintainers of **FastAPI**, **Uvicorn**, and other open-source libraries used in this project.



## Contributors 

