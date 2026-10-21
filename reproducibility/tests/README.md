# Tests

`test_reproducibility.py` contains artifact-level smoke tests:

1. checkpoint SHA256, key count (36), parameter count (9,026,563), `in_proj_weight` shape, no `pos_embed`
2. four-head strict load succeeds
3. eight-head + positional-embedding strict load fails (expected)
4. Random-Forest parameters (500 trees, depth 20, 512 features) — production joblibs load with a version warning; the stage2 family skips without the legacy env
5. linker-classifier load (with the documented numpy shim) — skips without the legacy env
6. external funnel counts (2,562 / 2,484 / 2,431 / 2,256)
7. split independence (scaffold/E3 compound-disjoint; random/target row-level; zero external overlap)
8. group-disjoint linker metrics reproduce (ChemBERTa)

Run: `python -m pytest tests/ -v`
Set `SYNGLUE_ROOT` (project) and `SYNGLUE_PY_ROOT` (models directory) if your layout differs.
