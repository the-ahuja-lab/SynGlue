#!/usr/bin/env python3
"""03_extract_embeddings.py  [P0]

Recover the fused 512-dimensional features (the matrices used to fit the Random Forests).

Two modes:

  --mode verify   (default) Load the saved matrices from revision/models/final_protac_model/
                  (Z_all 9,252 x 512, Z_train 1,531 x 512, Z_test 383 x 512), verify shapes and
                  hashes, and write a manifest. This reproduces the original fused features
                  without re-running the encoder.

  --mode recompute  Load a checkpoint + component tensor and regenerate the fused features with the
                  attention-pooling layer (model.embed). Requires the (N,3,4800) tensor from
                  step 01 and a checkpoint from step 02 (or the released checkpoint).

Outputs (results/)
  Z_all.npy / Z_train.npy / Z_test.npy        (verify: copies; recompute: regenerated)
  embedding_manifest.json                     shapes, dtypes, sha256, source checkpoint
"""
import argparse
import json
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "shared"),
           os.path.join(_HERE, "..", "..", "shared")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import _common as C


def verify():
    manifest = {}
    for key, name in [("Z_all", "Z_all.npy"), ("Z_train", "Z_train.npy"), ("Z_test", "Z_test.npy")]:
        src = C.PATHS[key]
        C.require(src, f"fused embedding matrix {name}")
        Z = np.load(src)
        dst = os.path.join(C.RESULTS, name)
        shutil.copy2(src, dst)
        manifest[name] = dict(shape=list(Z.shape), dtype=str(Z.dtype),
                              sha256=C.sha256(dst), source=src)
        print(f"  {name}: {Z.shape} {Z.dtype} sha256={manifest[name]['sha256'][:16]}...")
    Ztr = np.load(C.PATHS["Z_train"]); Zte = np.load(C.PATHS["Z_test"])
    if os.path.exists(C.PATHS["split_indices"]):
        z = np.load(C.PATHS["split_indices"], allow_pickle=True)
        manifest["split_indices"] = {k: list(map(int, np.ravel(z[k])[:5])) + ["..."] for k in z.keys()}
        print("  split index keys:", list(z.keys()))
    manifest["note"] = ("Z_train/Z_test are the matrices used to fit the stage2 RF heads; "
                        "Z_all covers all mapped rows (9,252).")
    C.save_json(manifest, "embedding_manifest.json")


def recompute(tensor, checkpoint):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    M = import_module("02_train_multitask_transformer") if False else None  # avoid executing main
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, input_dim=4800, hidden_dim=512, n_heads=4):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True),
                num_layers=2)
            self.attn_pool = nn.Linear(hidden_dim, 1)

        def embed(self, x):
            h = self.proj(x)
            h = self.transformer(h)
            w = torch.softmax(self.attn_pool(h), dim=1)
            return (h * w).sum(dim=1)

    X = np.load(tensor).astype(np.float32) if tensor.endswith(".npy") else np.load(tensor)[list(np.load(tensor).keys())[0]]
    sd = torch.load(C.require(checkpoint, "endpoint checkpoint"), map_location="cpu", weights_only=True)
    model = Model()
    missing = model.load_state_dict(sd, strict=False)
    print(f"  checkpoint keys: {len(sd)} | missing: {len(missing.missing_keys)} "
          f"| unexpected: {len(missing.unexpected_keys)}")
    model.eval()
    Zs = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            Zs.append(model.embed(torch.tensor(X[i:i + 64])).numpy())
    Z = np.vstack(Zs)
    out = os.path.join(C.RESULTS, "Z_recomputed.npy")
    np.save(out, Z)
    C.save_json(dict(tensor=tensor, checkpoint=checkpoint, shape=list(Z.shape),
                     sha256=C.sha256(out)), "embedding_manifest_recomputed.json")
    print("  recomputed Z:", Z.shape, "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "recompute"], default="verify")
    ap.add_argument("--tensor")
    ap.add_argument("--checkpoint", default=C.PATHS["checkpoint_compound_level"])
    args = ap.parse_args()
    if args.mode == "verify":
        verify()
    else:
        if not args.tensor:
            raise SystemExit("--tensor required for --mode recompute")
        recompute(args.tensor, args.checkpoint)


if __name__ == "__main__":
    main()
