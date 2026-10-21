#!/usr/bin/env python3
"""Convert LigBuilder cavity pharmacophore PDBs (5HXB GSPT1) to PGMG .posp format."""
import os

CAVITY_DIR = "/storage/savi/saveenas/Projects/Magnet/Gcoupler/GSPT1/cavity_files"
OUT_DIR = "/storage/savi/saveenas/Projects/SynGlue/revision/experiments/gcoupler_benchmark/results/warheads/pgmg/pharmacophores"
os.makedirs(OUT_DIR, exist_ok=True)

# PGMG types: AROM, HYBL, POSC, HACC, HDON, LHYBL
# LigBuilder types: N = h-bond donor, O = h-bond acceptor, C = hydrophobic center
MAP = {"N": "HDON", "O": "HACC", "C": "HYBL"}

for i in range(1, 10):
    src = os.path.join(CAVITY_DIR, "5HXB_chainA_clean_pharmacophore_%d.pdb" % i)
    if not os.path.exists(src):
        continue
    pts = []
    for line in open(src):
        if line.startswith("HETATM"):
            el = line[12:16].strip()
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            pts.append((el, x, y, z))
    # Deduplicate close points (<0.5 A)
    kept = []
    for p in pts:
        dup = False
        for k in kept:
            d = ((p[1]-k[1])**2 + (p[2]-k[2])**2 + (p[3]-k[3])**2) ** 0.5
            if d < 0.5:
                dup = True
                break
        if not dup:
            kept.append(p)
    if len(kept) > 8:
        kept = kept[:8]  # PGMG max 8 nodes
    if len(kept) < 2:
        continue
    dst = os.path.join(OUT_DIR, "GSPT1_cavity%d.posp" % i)
    with open(dst, "w") as f:
        for el, x, y, z in kept:
            t = MAP.get(el, "HYBL")
            f.write("%s %.4f %.4f %.4f\n" % (t, x, y, z))
    print(dst, "->", len(kept), "points:", [(MAP.get(p[0],"HYBL")) for p in kept])
