# Readme

# MetaboGlue: A Generative AI-Powered Polypharmacology Toolkit

**MetaboGlue** is a Python package designed to facilitate the decoding and designing of complex multi-targeting metabolites using generative AI techniques. It offers a comprehensive suite of tools for researchers and developers working on polypharmacology and drug discovery, including functionalities for metabolite prediction, multi-target interactions, and data visualization.

![Asset 2.png](Readme%2015fed9bbdafb8084911bccf84d0c90bb/Asset_2.png)

---

## Features

- **Generative AI Models**: leverage AI to design and optimize PROTACs and multi-targeting molecules.
- **Polypharmacology**:  Classify molecules based on type and target mapping.
- **Flexible API**: Easily integrate MetaboGlue into your workflows with a RESTful API powered by FastAPI.
- **Cheminformatics Support**: Use tools for molecular representation, manipulation, and analysis powered by RDKit.
- **Modular Design**: MetaboGlue is built with multiple modules to perform tasks like browsing, annotating, and mapping metabolites.

---

## Installation

To install **MetaboGlue** from PyPI, use the following command:

```python
pip install MetaboGlue
```

## Alternatively, you can clone the repository and install it locally:

```python
git clone https://github.com/yourusername/MetaboGlue.git
cd MetaboGlue
pip install .
```

## Usage

# Start the API server:

```python
uvicorn main:app –reload
The API will be available at http://localhost:8000.

You can interact with the API endpoints using Python’s requests library:

import requests

url = “http://localhost:8000/mapper/”
file_path = “input_file.csv”

with open(file_path, “rb”) as f:
response = requests.post(url, files={“input_file”: f})

if response.status_code == 200:
with open(“output_file.csv”, “wb”) as out_file:
out_file.write(response.content)
```

# Modules

### 1. **Browser**

- Browse through compound data and visualize results.
- Backend: A database containing terminal fragments, ligands, and targets.

### 2. **Mapper**

- Map queries to relevant compound data and retrieve matching ligands and fragments.

### 3. **Annotator**

- Annotate compounds with:
    - Molecular type
    - Target information
    - Functional group

### 4. **Warhead Mapper**

- Map potential warheads for specific targets during drug design workflows.

### 5. **Generator**

- Generate new metabolites based on input data.
- Includes:
    - **Optimizer**: Fine-tune generated structures.
    - **Scorer**: Rank generated molecules.

---

---

## Structure-Guided Workflow

MetaboGlue offers a two-part approach:

1. **Data-Driven**
    - Utilize structural databases to extract terminal fragments, ligands, and target mappings.
    - Map input molecules, annotate types and targets, and generate optimized molecules.
2. **Structure-Guided**
    - Input structures PDB (from databases like PDB ,).
    - Use the **GCoupler** module to synthesize molecules.
    - Employ warhead selection and scoring tools for optimization.

---

## Dependencies

- **fastapi** — API server
- **uvicorn** — ASGI server for FastAPI
- **requests** — HTTP requests
- **pandas** — Data manipulation
- **rdkit** — Cheminformatics toolkit
- **setuptools** — Python packaging
- **biopython** — Bioinformatics tasks
- **matplotlib** — Data visualization
- **scikit-learn** — Machine learning
- **seaborn** — Data visualization
- **scipy** — Scientific computing
- **tqdm** — Progress bars

---

---

## License

MetaboGlue is licensed under the MIT License. See [LICENSE](https://www.notion.so/saveenasolanki/LICENSE) for more details.

---

## Acknowledgments

- Thanks to the **RDKit team** for their powerful cheminformatics toolkit.
- Gratitude to the contributors and maintainers of **FastAPI**, **Uvicorn**, and other open-source libraries used in this project.

---

## Summary

MetaboGlue is a comprehensive toolkit that combines **data-driven** and **structure-guided** methodologies to design and analyze polypharmacological metabolites. It enables researchers to:

- Map and annotate metabolites with structural and functional insights.
- Optimize molecular generation workflows for drug discovery.
- Integrate easily into existing pipelines via a RESTful API.

With MetaboGlue, decoding multi-targeting metabolites has never been easier!