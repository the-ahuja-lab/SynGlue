#!/usr/bin/env python3
"""
Extract pocket residues from a PDB into a new PDB file (for DiffLinker).
"""
import argparse
from Bio.PDB import PDBParser, PDBIO, Select

class ResidueSelect(Select):
    def __init__(self, resis):
        self.resis = set(int(r) for r in resis)
    def accept_residue(self, residue):
        return residue.get_id()[1] in self.resis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--resis", required=True, help="space-separated residue numbers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    resis = args.resis.split()
    p = PDBParser(QUIET=True)
    struct = p.get_structure("p", args.pdb)
    io = PDBIO()
    io.set_structure(struct)
    io.save(args.out, ResidueSelect(resis))
    print(f"wrote {args.out} with residues {resis}")

if __name__ == "__main__":
    main()
