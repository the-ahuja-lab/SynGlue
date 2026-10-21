#!/usr/bin/env python3
"""Generate machine-readable evidence tables for the SynGlue final revision audit.
All rows encode verified measurements from this audit; statuses use the agreed vocabulary."""
import csv, os, json
OUT = "/storage/savi/saveenas/Projects/SynGlue/outputs/synglue_final_revision_evidence"
MR = f"{OUT}/machine_readable"
os.makedirs(MR, exist_ok=True)

def w(name, header, rows):
    p = f"{OUT}/{name}" if name.startswith(("00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15")) else f"{MR}/{name}"
    with open(p, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)
    print("wrote", p, len(rows), "rows")

# ---------------- 01 repository inventory ----------------
inv_hdr = ["artifact_name","location","function","version","checksum","source","input","output",
           "publication_mapping","reproduction_status","issue"]
inv = [
 ["multitask_transformer.pt","SynGlue_Py/models/","inference (frozen encoder)","see SHA256",
  "sha256:630783ce2f6f26e779e6aae004e0b6827135819ba3be3a2c16f458ea999d6e05","training notebook Attention-Copy1.ipynb cell 114",
  "3x4800 token tensor","fused 512-d representation","Fig 1e-g, Fig 2f/m, all endpoint results",
  "verified (loads strict with 4-head class; 36 keys; 9,026,563 params)","head count not stored in checkpoint"],
 ["rf_dc50.joblib","SynGlue_Py/models/","inference (DC50 regression)","sklearn<=1.0.2 pickle",
  "sha256:fc17095499b376dbee227e1021b497ea8af53ec3fdf068fcd38dfc7693907e10","unknown fitting script",
  "fused 512-d embeddings","DC50 (log10 nM)","endpoint tables","artifact verified (500 trees, depth 20, 512 features); fitting tensor UNRESOLVED","training matrix not deposited"],
 ["rf_dmax.joblib","SynGlue_Py/models/","inference (Dmax regression)","sklearn<=1.0.2 pickle",
  "sha256:5df8d3f0f5346f98792d61db1ea71cd050d325e0e3b5838090fdf853b99e4633","unknown fitting script",
  "fused 512-d embeddings","Dmax (%)","endpoint tables","artifact verified; fitting tensor UNRESOLVED","training matrix not deposited"],
 ["grover_fixed.pt","SynGlue_Py/models/","featuriser checkpoint","upstream GROVER",
  "sha256:6256228ac5b80ce9fd66dea80457c9428cdac64d254d946ea266ec94c4c65471","GROVER training",
  "SMILES","4800-d embeddings","all model inputs","not independently retrained","upstream commit not pinned"],
 ["linker_classifier.pkl","Reproducibility/Zenodo/.../06_Generative_Pipeline/models/","linker multiclass inference","sklearn legacy pickle",
  "md5:d5204351e9c42dbfbe1bc108a478d09f","PROTAC_Grover.ipynb cell 14","4800-d GROVER features",
  "4-class linker label","Fig 1g, Fig S4","identity verified (md5, 3 identical copies); load FAILED (pickle incompatibility)","no environment lock"],
 ["06_Generative_Pipeline_generator.py","Architecture_Code/","generation + scoring","public commit 2a8f57c",
  "md5:2b4a5ef195ebd9ce72d5baa6c2652f2e (public/Zenodo)","authors","warhead/E3 fragments",
  "candidate PROTACs + predictions","Methods; Fig 2","verified readable; scoring uses constant component tokens","local copy is git-dirty (pos_embed)"],
 ["05_Models_attention_dataset_builder.py","Architecture_Code/","component mapping + tensor build","public commit 2a8f57c",
  "unhashed","authors","PROTAC + component CSVs","X/y tensors","Methods","verified readable","first-match mapping defect"],
 ["05_Models_attention_model.py","Architecture_Code/","training (deposited)","public commit 2a8f57c",
  "unhashed","authors","y_dc50.npy only","single-head model","Methods","CONTRADICTED (single-task MSE; imports missing model_architecture)","non-runnable as deposited"],
 ["05_Models_model_config.py","Architecture_Code/","config (dead code)","public commit 2a8f57c",
  "unhashed","authors","none","none","Methods","CONTRADICTED (num_heads=8; never imported)","dead code"],
 ["05_Models_multi_class_linker.py","Architecture_Code/","linker classifier training","public commit 2a8f57c",
  "unhashed","authors","features_chemberta.csv","metrics + joblib","Methods; Fig S4","verified readable","selects by Test AUC"],
 ["reproduce_original_model.py","revision/","reproduction (masked multitask + RF)","local",
  "unhashed","authors","Upadted_PROTAC_features.csv","metrics","rebuttal reproduction","verified run: 0.682/0.574","omits 0.01 Dmax weight"],
 ["WP1_generalization_test.py","revision/","writes generalization_results.csv","local",
  "unhashed","authors","tensors","generalization_results.csv","rebuttal","writer identified; reproduction NOT demonstrated","provenance gap"],
 ["train_generalization_v4.py","revision/","v4 retrain (extras merged)","local",
  "unhashed","authors","X_{task}_pca.npy + extras","v4 models","rebuttal generalization","verified readable","19/14 external compounds leaked via extras"],
 ["ws2_synglue_wholeprotac.py","revision/","whole-PROTAC external scoring","local",
  "unhashed","authors","external SMILES","whole-PROTAC predictions","external benchmark","verified readable","repeat(GROVER(full),3) path"],
 ["Attention-Copy1.ipynb","server_53/.../Attention/","training notebook (executed)","executed run",
  "unhashed","authors","Upadted_PROTAC_features.csv + components","best_multitask_transformer.pt",
  "Methods; Fig 1e-g","verified: output SHA-identical to released checkpoint","training tensor 142,727 rows"],
 ["PROTAC_Grover.ipynb","Reproducibility/Savi/PROTAC/Protac_Grover/","linker classifier training","not in Zenodo inventory",
  "unhashed","authors","features","multiclass_linker_grover_GB.pkl","Fig 1g; Fig S4","verified via md5 identity","not deposited"],
 ["PROTAC_Mordred.ipynb","Reproducibility/Savi/PROTAC/PROTAC_Mordred/","comparison representation + CV","not in Zenodo inventory",
  "unhashed","authors","Mordred features","10-fold results","Fig S4","partially reproducible","not deposited"],
 ["Upadted_PROTAC_features.csv","server_53/.../Attention/data/","training table","working copy",
  "unhashed","MagnetDB/PROTAC-DB","raw labels","10,330 rows / 5,203 compounds","Methods","verified counts","not in public deposit"],
 ["Updated_warhead.csv","same","component library","working copy","unhashed","PROTAC-DB","-","1,104 rows / 365 IDs","Methods","verified","not in public deposit"],
 ["linker_updated_features.csv","same","component library","working copy","unhashed","PROTAC-DB","-","2,749 rows","Methods","verified","not in public deposit"],
 ["E3_updated_features.csv","same","component library","working copy","unhashed","PROTAC-DB","-","117 rows","Methods","verified","not in public deposit"],
 ["generalization_results.csv","revision/benchmarking/","reported split metrics","local",
  "unhashed","WP1_generalization_test.py","tensors","R2 table","manuscript/rebuttal","values 0.342/0.178; reproduction 0.682/0.574","provenance UNRESOLVED"],
 ["generalization_results_two_stage.csv","revision/benchmarking/","8-head two-stage metrics","local",
  "unhashed","two-stage pipeline","tensors","R2 table","rebuttal","verified readable","85,592 training rows (fan-out)"],
 ["oriented_metrics.csv","revision/rebuttal_analyses/11_.../outputs/","comparator AUROC/EF","local",
  "unhashed","analysis.py","predictions","AUROC table","Fig 8/S9","verified readable","TACK n label inverted"],
 ["tool_metrics.csv","revision/rebuttal_analyses/08_.../outputs/","pre-orientation metrics","local",
  "unhashed","analysis.py","predictions","rho/AUC","rebuttal","verified readable","AUC <0.5 due to orientation artifact"],
]
w("01_REPOSITORY_INVENTORY.csv", inv_hdr, inv)

# ---------------- 03 leakage audit ----------------
lh = ["evaluation","split_or_population","n_train_rows","n_train_unique_compounds","n_val_rows",
      "n_test_rows","n_test_unique_compounds","exact_smiles_overlap_train_test","shared_compound_id_overlap",
      "scaffold_overlap","endpoint_label_reuse","encoder_fitted_within_cv","test_labels_in_selection",
      "status","evidence"]
leak = [
 ["Original reproduction (random 80/20)","random",737,"UNRESOLVED",171,228,"UNRESOLVED",0,0,"UNRESOLVED","no",
  "no (validation early stopping)","no","VERIFIED for rows; compound-level UNRESOLVED","generalization_results.csv"],
 ["v4 DC50 retrain","random/scaffold/target/E3",2104,"UNRESOLVED",0,0,"UNRESOLVED","19 unique (32 rows)","19","UNRESOLVED","YES (19 compounds)",
  "no","no","CONTRADICTED: external compounds in training","metadata_dc50_extra.csv overlap"],
 ["v4 Dmax retrain","random/scaffold/target/E3",1228,"UNRESOLVED",0,0,"UNRESOLVED","14 unique (31 rows)","14","UNRESOLVED","YES (14 compounds)",
  "no","no","CONTRADICTED: external compounds in training","metadata_dmax_extra.csv overlap"],
 ["External benchmark (activity)","frozen external",0,0,0,2562,"2562",0,0,"n/a","no",  "n/a","no","VERIFIED (labels excluded); 19/14 compounds affected v4 only","final_external_tack_chiral.csv"],
 ["External benchmark (strict OOD)","frozen external",0,0,0,2256,"2256",0,0,"n/a","no","n/a","no","VERIFIED","tmax_chiral<0.70"],
 ["Linker classifier","80/20 stratified",1856,"UNRESOLVED (846 IDs overall)",0,464,"UNRESOLVED","UNRESOLVED (repeated structures)","UNRESOLVED","UNRESOLVED","YES (class labels)",
  "no","YES (max Test AUC)","CONTRADICTED protocol: test-selected","05_Models_multi_class_linker.py:204"],
 ["Linker classifier 10-fold (Grover)","10x repeated fixed test",1856,"UNRESOLVED",0,464,"UNRESOLVED","0 (fixed test)","0","UNRESOLVED","YES",
  "no","YES","repeated test evaluation, not CV","10_fold_results.csv median 0.7975"],
 ["Linker classifier (Mordred Final_10_CV_1)","10-fold held-out",1755,"UNRESOLVED",0,439,"UNRESOLVED","0","0","UNRESOLVED","YES",
  "no (folds inside train)","no (but post-hoc hyperparameters)","cleaner; not nested","final_test_results.csv 0.73272"],
 ["Target-held-out (v4 DC50)","target-held",1419,"UNRESOLVED",251,434,"UNRESOLVED","0","0","UNRESOLVED","no","no","no","VERIFIED rows; compound-level UNRESOLVED","generalization_summary_v4.csv"],
 ["Dmax target-held-out (v4)","target-held",895,"UNRESOLVED",158,175,"UNRESOLVED","0","0","UNRESOLVED","no","no","no","VERIFIED rows","generalization_summary_v4.csv"],
]
w("03_TRAIN_EVAL_LEAKAGE_AUDIT.csv", lh, leak)

# ---------------- 04 dataset reconciliation ----------------
dh = ["item","manuscript_claim","reconstructed_value","unit","rule_or_derivation","status","source"]
ds = [
 ["PROTAC-DB 3.0 entries","9,380","9,380","record","rows in PROTAC_DB_with_features.csv","VERIFIED","PROTAC_DB_with_features.csv"],
 ["PROTAC-DB unique SMILES","6,111","6,110","compound","RDKit canonical dedup on Smiles","CONTRADICTED (off by one)","T1 reconstruction"],
 ["Training table rows","10,330","10,330","row","Upadted_PROTAC_features.csv","VERIFIED","wc -l 10,331"],
 ["Unique training PROTACs","5,203","5,203","compound","canonical SMILES","VERIFIED","reconstruction"],
 ["DC50 labelled records","2,446","2,446","record","2,468 non-null - 22 literal N.D.","VERIFIED value / rule CONTRADICTED (417 censored, 417 multi-valued, 774 distinct)","T1"],
 ["Dmax labelled records","1,728","1,728 (raw) / 1,364 (no N.D.) / 1,062 (numeric only)","record","raw non-null includes 364 N.D.","CONTRADICTED rule","T1"],
 ["Unique linkers","2,749","2,749 IDs / 2,494 structures","linker","dedup on Linker ID vs SMILES","VERIFIED (ID) / ambiguous (structure)","linker_updated_features.csv"],
 ["Linker-classification rows","2,320 PROTACs","2,320 rows, 846 IDs, 675 SMILES","row","featuresChemberta.csv","CONTRADICTED unit","T1"],
 ["Mapped PROTACs (builder)","not stated","9,252","compound","builder output","VERIFIED","notebook log"],
 ["Tensor rows","not stated","142,727","row","merge fan-out on compound IDs","VERIFIED","notebook log"],
 ["Unique compounds in tensor","not stated","4,585","compound","reconstruction","VERIFIED (reconstruction)","T1/T2"],
 ["DC50-valid tensor rows","not stated","12,843","row","log","VERIFIED","notebook log"],
 ["Dmax-valid tensor rows","not stated","8,520","row","log","VERIFIED","notebook log"],
 ["Unique compounds with DC50","not stated","1,145","compound","reconstruction","VERIFIED (reconstruction)","T2"],
 ["External source (TACK)","3,514","3,514","compound","canonical dedup of TACK parquet","VERIFIED","per_source_vs_DB3.csv"],
 ["Assembled external pool","2,930","2,930","compound","final_dataset - PROTAC-DB, dedup","VERIFIED","COUNT_TRACE.md"],
 ["Exact-overlap-free set","2,562","2,562","compound","-952 exact matches to 6,110","VERIFIED","final_external_tack_chiral.csv"],
 ["Tanimoto<0.90","2,484","2,484","compound","chiral Morgan","VERIFIED","same"],
 ["Tanimoto<0.85","2,431","2,431","compound","chiral Morgan","VERIFIED","same"],
 ["Tanimoto<0.70","2,256","2,256","compound","chiral Morgan","VERIFIED","same"],
 ["Patent source rows","63,136","63,136","row","official PROTAC-PatentDB","VERIFIED","PROTAC_Patent_Compounds_full.csv"],
 ["Patent unique","57,533","57,533","compound","canonical dedup","VERIFIED","patent_full_check.csv"],
 ["Patent Tanimoto<0.70","43,891","43,891","compound","achiral Morgan","VERIFIED","patent_full_check.csv"],
]
w("04_DATASET_RECONCILIATION.csv", dh, ds)

# ---------------- 07 external benchmark ----------------
eh = ["model","n_evaluated","auroc_le_10nM","auroc_le_100nM","auroc_le_1uM","spearman_rho","leakage_rule","status","note"]
eb = [
 ["SynGlue 3-token",71,0.6963,0.7556,"n/a (insufficient positives)",0.4391,"exact-in-SynGlue removed; 151-set only","VERIFIED","reported 0.76 at <=100nM matches"],
 ["SynGlue whole-PROTAC",1700,0.5111,0.6170,0.6659,0.0548,"exact-in-SynGlue removed","VERIFIED","0.84 at <=1uM NOT reproduced"],
 ["TACK-XGB DC50",1481,"UNRESOLVED","0.9390","0.9903","0.8506","non-held-out (label inverted)","CONTRADICTED protocol","109 exact-in-SynGlue included"],
 ["TACK-MLP DC50",1481,"UNRESOLVED","0.9238","0.9875","0.7881","non-held-out (label inverted)","CONTRADICTED protocol","same"],
 ["PROTAC-DC50",71,"UNRESOLVED",0.6717,"n/a",0.2310,"exact-in-SynGlue removed","VERIFIED","151-set only"],
 ["Morgan-RF",1809,"UNRESOLVED",0.7471,0.8123,0.3425,"none","VERIFIED","baseline"],
 ["Tanimoto-NN",1809,"UNRESOLVED",0.6880,0.7497,0.1343,"none","VERIFIED","baseline"],
 ["PDP",1809,"UNRESOLVED",0.6499,0.8560,0.3060,"none","VERIFIED","classifier"],
 ["PDP-XGB",1809,"UNRESOLVED",0.6481,0.8261,0.3219,"none","VERIFIED","classifier"],
 ["PROTAC-STAN",1602,"UNRESOLVED",0.5434,0.7920,0.0805,"exact-in-STAN removed","VERIFIED","classifier"],
]
w("07_EXTERNAL_BENCHMARK_RECONCILIATION.csv", eh, eb)

# ---------------- 08 figure audit ----------------
fh = ["figure_panel","claim_or_content","source_available","regenerated","issue","required_action","status"]
fa = [
 ["Fig 1e-g","architecture of component-aware regression + linker classification","code only","no","two systems shown; 4 vs 8 head ambiguity","redraw with 4-head released model and separate classifier","DOCUMENTED_ONLY"],
 ["Fig S4","linker classification ROC/AUC, 10-fold","notebooks + metrics","partial","0.80 is test-selected; 10-fold is repeated fixed-test","relabel or rerun sealed/nested","CONTRADICTED"],
 ["Fig S5","Transformer architecture (4 heads, 2 layers, no pos-embed)","checkpoint + notebook","yes (spec)","8-head two-stage variant also exists","label variants explicitly","CORRECTED (text)"],
 ["Fig S6","external data funnel","scripts + CSVs","yes (counts)","2,930 described as overlap-excluded","restate: pool pre-overlap; 2,562/2,256 are filtered","CORRECTED"],
 ["Fig S7","random/scaffold/target/clinical/external metrics","partial","no","0.76/0.57/0.83 not found in result files","locate source or remove","UNRESOLVED"],
 ["Fig S8","model comparison denominators and GSPT1 benchmark","partial","no","denominators differ per comparator","report per-comparator n","UNRESOLVED"],
 ["Fig S9","external comparator AUROCs","oriented_metrics.csv","yes","SynGlue 0.84 at <=1uM not reproduced","correct or qualify","CONTRADICTED"],
 ["Fig 2f","predicted DC50/Dmax for BRD4 cmp1 vs cmp2","figure source not found","no","example artifact gives cmp2 lower DC50","deposit source values","UNRESOLVED"],
 ["Fig 2m","simulation/outputs provenance","not found","no","no source data","deposit or remove","UNRESOLVED"],
 ["Fig 3a","cellular GI50","experimental data","n/a","not computational","none","VERIFIED (experimental, out of scope)"],
]
w("08_FIGURE_TABLE_AUDIT.csv", fh, fa)

# ---------------- machine-readable extras ----------------
w("artifact_hashes.csv", ["artifact","sha256_or_md5","type","path"],
  [["multitask_transformer.pt","630783ce2f6f26e779e6aae004e0b6827135819ba3be3a2c16f458ea999d6e05","sha256","SynGlue_Py/models/"],
   ["rf_dc50.joblib","fc17095499b376dbee227e1021b497ea8af53ec3fdf068fcd38dfc7693907e10","sha256","SynGlue_Py/models/"],
   ["rf_dmax.joblib","5df8d3f0f5346f98792d61db1ea71cd050d325e0e3b5838090fdf853b99e4633","sha256","SynGlue_Py/models/"],
   ["grover_fixed.pt","6256228ac5b80ce9fd66dea80457c9428cdac64d254d946ea266ec94c4c65471","sha256","SynGlue_Py/models/"],
   ["linker_classifier.pkl","d5204351e9c42dbfbe1bc108a478d09f","md5","Reproducibility/Zenodo/"],
   ["06_Generative_Pipeline_generator.py","2b4a5ef195ebd9ce72d5baa6c2652f2e","md5","Architecture_Code/ (public/Zenodo)"]])

w("comparator_n.csv", ["comparator","n","leakage_control","status"],
  [["SynGlue 3-token",71,"exact-in-SynGlue; decomposed 151-set","VERIFIED"],
   ["SynGlue whole-PROTAC",1700,"exact-in-SynGlue","VERIFIED"],
   ["TACK-XGB",1481,"non-held-out (label inverted)","CONTRADICTED"],
   ["TACK-MLP",1481,"non-held-out (label inverted)","CONTRADICTED"],
   ["PROTAC-DC50",71,"exact-in-SynGlue; 151-set","VERIFIED"],
   ["PROTAC-STAN",1602,"exact-in-STAN","VERIFIED"],
   ["Morgan-RF",1809,"none","VERIFIED"],
   ["Tanimoto-NN",1809,"none","VERIFIED"],
   ["PDP",1809,"none","VERIFIED"],
   ["PDP-XGB",1809,"none","VERIFIED"]])

json.dump({"union_filter_intersection":13,"without_151_set_models":289,"coverage_only":71,
           "tack_reported_n":1481,"tack_held_out_n":328,"tack_exact_in_synglue":109,
           "note":"13 is prediction-availability under union filters, not a common evaluation cohort"},
          open(f"{MR}/intersection.json","w"), indent=2)
print("machine-readable extras written")
