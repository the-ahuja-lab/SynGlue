# setup_databases.sh

#!/bin/bash

echo "🔷 Setting up SynGlue Database Structure..."

# ==============================
# 1. Create folder structure
# ==============================

mkdir -p 00_Database/{raw,processed,final}

cd 00_Database/raw

echo "📂 Created folders: raw / processed / final"

# ==============================
# 2. DrugBank
# ==============================

mkdir -p DrugBank
cd DrugBank

echo "⬇ Downloading DrugBank..."

# NOTE: requires login → placeholder
echo "⚠ DrugBank requires login. Please download manually from:"
echo "https://go.drugbank.com/releases/5-1-10"

cd ..

# ==============================
# 3. BindingDB
# ==============================

mkdir -p BindingDB
cd BindingDB

echo "⬇ Downloading BindingDB..."

wget -c https://bindingdb.org/bind/BindingDB_All.tsv.zip

cd ..

# ==============================
# 4. ChEMBL
# ==============================

mkdir -p ChEMBL
cd ChEMBL

echo "⬇ Downloading ChEMBL v33..."

wget -c https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz

cd ..

# ==============================
# 5. STITCH
# ==============================

mkdir -p STITCH
cd STITCH

echo "⬇ Downloading STITCH..."

wget -c http://stitch.embl.de/download/protein_chemical.links.v5.0.tsv.gz

cd ..

# ==============================
# 6. BioSNAP
# ==============================

mkdir -p BioSNAP
cd BioSNAP

echo "⬇ Downloading BioSNAP datasets..."

wget -c https://snap.stanford.edu/biodata/datasets/10016/10016-ChG-InterDecagon.html
wget -c https://snap.stanford.edu/biodata/datasets/10001/10001-ChCh-Miner.html
wget -c https://snap.stanford.edu/biodata/datasets/10015/10015-ChG-TargetDecagon.html

cd ..

# ==============================
# 7. Small Molecule Suite
# ==============================

mkdir -p Small_Molecule_Suite
cd Small_Molecule_Suite

echo "⬇ Downloading Small Molecule Suite..."

wget -c https://lsp.connect.hms.harvard.edu/smallmoleculesuite/

cd ..

# ==============================
# 8. PROTAC-DB
# ==============================

mkdir -p PROTAC_DB
cd PROTAC_DB

echo "⬇ Downloading PROTAC-DB..."

wget -c https://cadd.zju.edu.cn/protacdb/

cd ..

# ==============================
# 9. PROTACpedia
# ==============================

mkdir -p PROTACpedia
cd PROTACpedia

echo "⬇ Downloading PROTACpedia..."

wget -c https://protacpedia.weizmann.ac.il/ptcb/download

cd ..

# ==============================
# DONE
# ==============================

cd ../../

echo "✅ All downloads attempted."
echo "⚠ Some datasets require manual download (DrugBank, portals)."

echo "📁 Structure ready:"
echo "00_Database/raw/"
echo "00_Database/processed/"
echo "00_Database/final/"

# Runnig commands
# chmod +x setup_databases.sh
# ./setup_databases.sh