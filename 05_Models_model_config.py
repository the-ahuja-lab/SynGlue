# =========================================================
# MODEL CONFIGURATION (REPRODUCIBLE)
# =========================================================

MODEL_CONFIG = {
    "input_dim": 4800,        # GROVER embedding size per component
    "proj_dim": 512,          # projection dimension
    "num_heads": 8,           # attention heads
    "num_layers": 2,          # transformer layers
    "ff_dim": 2048,           # feedforward dimension
    "dropout": 0.1,
    "regressor_hidden": 256,
    "regressor_dropout": 0.2
}