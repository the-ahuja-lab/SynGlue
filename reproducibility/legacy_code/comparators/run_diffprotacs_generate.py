#!/usr/bin/env python3
"""
DiffPROTACs driver for GSPT1 (warhead + CRBN ligand -> PROTAC linker diffusion).
Single-GPU inference with horovod mock (same approach as run_sample.py but with
GSPT1 fragments + custom linker length).
"""
import argparse
import os
import sys
import types

# horovod mock
mock_hvd = types.ModuleType('horovod')
mock_hvd_torch = types.ModuleType('horovod.torch')

class MockHVD:
    @staticmethod
    def local_rank():
        return 0
    @staticmethod
    def rank():
        return 0
    @staticmethod
    def size():
        return 1
    @staticmethod
    def init():
        pass
    @staticmethod
    def allreduce(tensor, name=None):
        return tensor

mock_hvd.torch = MockHVD()
sys.modules['horovod'] = mock_hvd
sys.modules['horovod.torch'] = MockHVD()

DIFFPROTACS = "/storage/savi/saveenas/Projects/SynGlue/validation_tools/DiffPROTACs"
sys.path.insert(0, DIFFPROTACS)
os.chdir(DIFFPROTACS)

from sample import create_argparser, main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join(DIFFPROTACS, "checkpoints/protacs_best.ckpt"))
    ap.add_argument("--xyz_path", required=True)
    ap.add_argument("--linker_len", type=int, default=8)
    ap.add_argument("--name", default="GSPT1")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=50)
    args = ap.parse_args()

    sys.argv = [
        "sample.py",
        "--model_path", args.model_path,
        "--xyz_path", args.xyz_path,
        "--linker_len", str(args.linker_len),
        "--name", args.name,
        "--output_dir", args.output_dir,
        "--n_samples", str(args.n_samples),
        "--batch_size", str(args.batch_size),
        "--num_workers", "2",
    ]
    a = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(a)
    print("DiffPROTACs GSPT1 generation complete ->", args.output_dir)
