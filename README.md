# SynGlue: A Generative AI Toolkit for PROTACs Decoding and Designing

**SynGlue** is a powerful Python-based platform for the generation, analysis, and optimization of PROTACs (Proteolysis Targeting Chimeras), and multitarget molecules. Built for researchers in cheminformatics, structural biology, and drug discovery. SynGlue uses data-driven AI to accelerate the design of small molecules, predict degradation potency (DC₅₀, Dmax), and guide rational linker selection. SynGlue offers an end-to-end suite of tools for generating PROTACs and their prioritising.

<br>
<div align="center">
<img src="images/Asset 2.png" alt="SynGlue architecture for PROTACs and molecular glue design via data-driven and structure-guided methods" ></div>
<br>

<div align="left">

<p>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9-blue">
  <img src="https://img.shields.io/badge/pypi-v0.1.6-orange">
  <img src="https://img.shields.io/conda/vn/conda-forge/YOUR_PACKAGE">
  <a href="https://colab.research.google.com/drive/1k3UyoqYU_zw6_GbdeaARe155dCi_JO6Q?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
</p>

</div>

---

## 🚀 Features

* **Generative AI Models** for PROTAC and multitarget molecule design
* **TRIE-based fragment storage algorithm** for fast retrieval
* **Polypharmacology classification**
* **REST API integration via SynGlue client**
* **RDKit-powered cheminformatics utilities**

---

# Installation

### 📦 Install API Client

```bash
pip install synglue requests
```

### 🧩 Clone Repository (Optional)

```bash
git clone https://github.com/the-ahuja-lab/SynGlue.git
cd SynGlue
```

---

# 🔌 SynGlue API Usage

```python
from synglue import SynGlue

client = SynGlue()

# Health check
print(client.health_check())
```

---

## 🚀 Design Workflow

```python
# Submit design job
design_result = client.submit_design(target="EGFR", threshold=80)
print(design_result)

# Check status
job_id = design_result["job_id"]
print(client.design_status(job_id))

# client.download_design(job_id, "design_results.zip")
```

---

## 🔍 Screening Workflow (List)

```python
molecules = [
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "Imatinib", "smiles": "CC1=CC=CC=C1"}
]

screen_result = client.submit_screen(molecules=molecules)
print(screen_result)

job_id = screen_result["job_id"]
print(client.screen_status(job_id))

# client.download_screen(job_id, "screen_results.zip")
```

---

## 📂 Screening Workflow (CSV)

```python
csv_path = "query.csv"

screen_result = client.submit_screen_csv(csv_path)
print(screen_result)

job_id = screen_result["job_id"]
print(client.screen_status(job_id))
```

---

# 📡 API Endpoints (Client Methods)

### 🔹 Core

* `health_check()`

### 🔹 Design

* `submit_design(target, threshold=75.0)`
* `design_status(job_id)`
* `download_design(job_id, out_path)`

### 🔹 Screening

* `submit_screen(molecules)`
* `submit_screen_csv(csv_path)`
* `screen_status(job_id)`
* `download_screen(job_id, out_path)`

---

## SynGlue Workflow

SynGlue offers a two-part approach:

* **Data-Driven**

  * Map input molecules, annotate types and targets, and generate optimized molecules.

* **Structure-Guided**

  * Use structural inputs (PDB, AlphaFold, Rosetta)
  * Perform warhead mapping and molecule generation

---

## 📊 Type Analysis

<br>
<img src="images/Asset_1.png" alt="Type Analysis" style="width: 35%; max-width: 300px; height: auto;">
<br>

| Classification               | Type   |
| ---------------------------- | ------ |
| **Monovalent, Monotarget**   | Type 1 |
| **Monovalent, Multitarget**  | Type 2 |
| **Multivalent, Monotarget**  | Type 3 |
| **Multivalent, Multitarget** | Type 4 |

---

## Tutorials

| Tutorial          | Description               | Colab Link                                                                                             |
| ----------------- | ------------------------- | ------------------------------------------------------------------------------------------------------ |
| PROTAC Generation | Generate PROTAC molecules | [Open in Colab](https://colab.research.google.com/drive/1k3UyoqYU_zw6_GbdeaARe155dCi_JO6Q?usp=sharing) |
| Target Mapping    | Screen and map targets    | [Open in Colab](https://colab.research.google.com/drive/1WgG_T-rD5sODGFpq9GQzi7vXctpq5neo?usp=sharing) |

---

## Summary

SynGlue enables:

* PROTAC design and prioritization
* Target mapping and screening
* Integration into automated drug discovery pipelines

---

## Dependencies

* Python ≥ 3.7
* requests

```bash
pip install requests
```

---

## License

SynGlue is licensed under the MIT License. See [LICENSE](https://www.notion.so/saveenasolanki/LICENSE) for more details.
