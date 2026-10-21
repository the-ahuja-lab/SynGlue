#!/usr/bin/env python3
"""DiffPROTACs single-GPU sampler (horovod mocked) for WS2 GSPT1 benchmark."""
import sys, types, os

mock = types.ModuleType('horovod')
class HVD:
    @staticmethod
    def local_rank(): return 0
    @staticmethod
    def rank(): return 0
    @staticmethod
    def size(): return 1
    @staticmethod
    def init(): pass
    @staticmethod
    def allreduce(t, name=None): return t
mock.torch = HVD()
sys.modules['horovod'] = mock
sys.modules['horovod.torch'] = HVD()

BASE = "/storage/savi/saveenas/Projects/SynGlue/validation_tools/DiffPROTACs"
OUT = "/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws2_gen/diffprotacs_out"
os.makedirs(OUT, exist_ok=True)
os.chdir(BASE)
sys.path.insert(0, BASE)

sys.argv = ['sample.py',
    '--model_path', os.path.join(BASE, 'checkpoints/protacs_best.ckpt'),
    '--xyz_path', '/storage/savi/saveenas/Projects/SynGlue/revision/benchmarking/ws2_gen/inputs/fragments.xyz',
    '--linker_len', '8',
    '--name', 'GSPT1_CRBN',
    '--output_dir', OUT,
    '--n_samples', '100',
    '--batch_size', '50',
    '--num_workers', '4']
exec(open(os.path.join(BASE, 'sample.py')).read())
print("DiffPROTACs sampling complete")
