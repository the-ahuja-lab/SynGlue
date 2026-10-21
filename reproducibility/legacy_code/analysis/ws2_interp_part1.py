#!/usr/bin/env python3
"""
Interpretability — Part 1: attention weights + component-ablation representations.
For the 151 decomposed external compounds (3-token input):
  - extract learned attention weights over [warhead, linker, E3]
  - compute fused Z for full input and for each zeroed-out component (warhead/linker/E3)
Saves Z matrices + attention for Part 2 (RF predictions in rf_predict env).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch

REV = "/storage/savi/saveenas/Projects/SynGlue/revision"
TIR = os.path.join(REV, "benchmarking", "ws1", "tiers")
SYNGLUE = "/storage/savi/saveenas/Projects/SynGlue_Py"
OUT = os.path.join(REV, "benchmarking", "ws1", "tiers", "interp")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REV)
from reproduce_original_model import MultiTaskProtacModel

# 151 decomposed set + component embeddings (cached in ws1 run)
sc = pd.read_csv(os.path.join(TIR, "tier_scoring_set.csv"))
def load_emb(tag):
    fps = np.load(os.path.join(TIR, "emb", f"{tag}_fps.npz"))["fps"][:, :4800]
    uniq = sorted(sc[f"{tag}_smiles"].dropna().unique())
    assert len(uniq) == fps.shape[0], f"{tag}: {len(uniq)} vs {fps.shape[0]}"
    return {s: fps[i] for i, s in enumerate(uniq)}
Wm, Lm, Em = load_emb("warhead"), load_emb("linker"), load_emb("e3")

X = np.zeros((len(sc), 3, 4800), dtype=np.float32)
for i, r in sc.iterrows():
    X[i, 0] = Wm[r["warhead_smiles"]]; X[i, 1] = Lm[r["linker_smiles"]]; X[i, 2] = Em[r["e3_smiles"]]

model = MultiTaskProtacModel(input_dim=4800)
model.load_state_dict(torch.load(os.path.join(SYNGLUE, "models", "multitask_transformer.pt"),
                                 map_location="cpu", weights_only=True))
model.eval()

with torch.no_grad():
    Xt = torch.tensor(X, dtype=torch.float32)
    h = model.proj(Xt); h = model.transformer(h)
    attn = torch.softmax(model.attn_pool(h), dim=1).numpy()   # (n,3)
    Z_full = (h * torch.softmax(model.attn_pool(h), dim=1)).sum(dim=1).numpy()

# ablations: zero one token slot
abl = {}
for tag, idx in [("warhead", 0), ("linker", 1), ("e3", 2)]:
    Xa = X.copy(); Xa[:, idx, :] = 0.0
    with torch.no_grad():
        ha = model.proj(torch.tensor(Xa, dtype=torch.float32))
        ha = model.transformer(ha)
        attna = torch.softmax(model.attn_pool(ha), dim=1)
        Za = (ha * attna).sum(dim=1).numpy()
    abl[tag] = Za

np.save(os.path.join(OUT, "Z_full.npy"), Z_full)
for tag, Za in abl.items():
    np.save(os.path.join(OUT, f"Z_abl_{tag}.npy"), Za)
pd.DataFrame(attn.reshape(-1, 3), columns=["warhead", "linker", "e3"]).to_csv(os.path.join(OUT, "attention_weights.csv"), index=False)
print("attention mean:", attn.mean(axis=0))
print("attention std:", attn.std(axis=0))
print("saved Z_full + Z_abl_* + attention_weights.csv")
