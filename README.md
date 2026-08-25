
> **Designing molecules that control protein fate.** SynGlue is a computational platform for
> decoding and designing PROTACs generative AI, interaction modelling, and degradation
> prediction in one end-to-end toolkit.

---

## Scientific question

*Why do apparently similar PROTACs produce very different degradation outcomes — and how can
we design the molecule that creates the right interaction?*

## Concept

<div align="center">
  <img src="images/Asset 2.png" alt="SynGlue architecture for PROTAC design via data-driven and structure-guided methods">
</div>

## SynGlue

SynGlue approaches PROTAC design as a coupled molecular-design problem:

> Target ligand + E3-ligase ligand + linker + ternary-complex geometry + degradation behaviour

rather than treating the warhead as the sole determinant of degrader activity. The platform
supports **molecular generation, linker reasoning, degradation modelling (DC₅₀, Dmax) and
structure-informed prioritisation** for PROTACs and multitarget molecules.

Two complementary workflows:

- **Data-driven** — map input molecules, annotate types and targets, generate optimised molecules
- **Structure-guided** — use structural inputs (PDB, AlphaFold, Rosetta) for warhead mapping and molecule generation

## Main methodological contributions

- **Generative AI models**  for de novo PROTAC and multitarget molecular design
- **Integrated warhead–E3** ligand-linker design and optimization
- **TRIE-based fragment storage** and rapid molecular retrieval
- **Polypharmacology-aware** — Type 1-4 molecular classification
- **Machine-learning** prediction of degradation activity, DC50 and Dmax
- **Integrated ADMET**, drug-likeness and multi-objective candidate ranking
- End-to-end deployable platform with **RDKit utilities**, **REST API** and **Python client**


## Benchmark & validation

Predictions of degradation potency (DC50, Dmax) and design-prioritisation benchmarks accompany
the forthcoming manuscript; the platform ships with interactive Colab tutorials for
reproducing the design and screening workflows.

## Installation

```bash
pip install synglue requests
```

Optional: clone the repository for the full source.

## Quick start

```python
from synglue import SynGlue

client = SynGlue()
print(client.health_check())

# Submit a design job
design_result = client.submit_design(target="EGFR", threshold=80)
job_id = design_result["job_id"]
print(client.design_status(job_id))
```

## Examples

**Screen molecules (list):**

```python
molecules = [
    {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    {"name": "Imatinib", "smiles": "CC1=CC=CC=C1"},
]
screen_result = client.submit_screen(molecules=molecules)
job_id = screen_result["job_id"]
print(client.screen_status(job_id))
```

**Screen molecules (CSV):**

```python
screen_result = client.submit_screen_csv("query.csv")
job_id = screen_result["job_id"]
print(client.screen_status(job_id))
```


## Reproducibility

- [PROTAC generation tutorial — Colab](https://colab.research.google.com/drive/1k3UyoqYU_zw6_GbdeaARe155dCi_JO6Q?usp=sharing)
- [Target mapping & screening tutorial — Colab](https://colab.research.google.com/drive/1WgG_T-rD5sODGFpq9GQzi7vXctpq5neo?usp=sharing)

## Citation

If you use SynGlue in your research, please cite:

Solanki S, Mohanty SK, Satija S, Chauhan S, Bandaru NVMR, Dukare S, Tiwari NK, Kumar RN, Aravind AB, Mukherjee S, Chikkanna D, Balasubramanian WR, Sammeta SR, Gautam V, Arora S, Kumar S, Duari S, Sharma A, Shome R, Sengupta D, Abbineni C, Samajdar S, Ahuja G.
Generative AI Framework SynGlue for the Rational Design of Clinically Relevant Protein Degraders.
bioRxiv (2025), 2025.08.28.672835.
https://doi.org/10.1101/2025.08.28.672835

## License

MIT — see [`LICENSE.txt`](LICENSE.txt).
