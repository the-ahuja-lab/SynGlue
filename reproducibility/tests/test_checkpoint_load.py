"""Checkpoint identity and architecture tests."""
import os, hashlib, pytest
import _common as C

EXPECTED_SHA = "630783ce2f6f26e779e6aae004e0b6827135819ba3be3a2c16f458ea999d6e05"

def test_released_checkpoint_hash_params_keys():
    import torch
    p = C.PATHS["checkpoint_released"]
    if not os.path.exists(p):
        pytest.skip("released checkpoint not present")
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == EXPECTED_SHA
    sd = torch.load(p, map_location="cpu", weights_only=True)
    assert len(sd) == 36
    assert sum(v.numel() for v in sd.values()) == 9_026_563
    assert tuple(sd["transformer.layers.0.self_attn.in_proj_weight"].shape) == (1536, 512)
    assert not any("pos_embed" in k for k in sd)

def test_four_head_class_loads_strictly():
    import torch, torch.nn as nn
    p = C.PATHS["checkpoint_released"]
    if not os.path.exists(p):
        pytest.skip("released checkpoint not present")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    class M4(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4800, 512)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True), num_layers=2)
            self.attn_pool = nn.Linear(512, 1)
            self.head_dc50 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
            self.head_dmax = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    M4().load_state_dict(sd, strict=True)

def test_eight_head_posembed_class_fails():
    import torch, torch.nn as nn
    p = C.PATHS["checkpoint_released"]
    if not os.path.exists(p):
        pytest.skip("released checkpoint not present")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    class M8(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4800, 512)
            self.pos_embed = nn.Embedding(3, 512)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True), num_layers=3)
            self.attn_pool = nn.Linear(512, 1)
            self.head_dc50 = nn.Linear(512, 1); self.head_dmax = nn.Linear(512, 1)
    with pytest.raises(RuntimeError):
        M8().load_state_dict(sd, strict=True)
