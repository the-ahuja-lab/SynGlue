"""masked_multitask_trainer.py

Extracted verbatim (source only, no outputs) from the executed training cell
`Attention-Copy1.ipynb`, cell 114 — the trainer that produced the released
checkpoint `multitask_transformer.pt` (4 heads, 2 layers, no positional embedding;
masked MSE with L = L_DC50 + 0.01*L_Dmax; AdamW lr 1e-4; batch 64/128; 100 epochs;
early-stopping patience 15; 80/20 split, seed 42).

Paths in the original cell are absolute; edit DATA paths before running.
"""

# =============================================================================
# SYNGLUE MULTI-TASK TRAINING PIPELINE (DC50 & Dmax)
# =============================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# =============================================================================
# 1. DATA INGESTION & NORMALIZATION (Your exact logic)
# =============================================================================
def load_full_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"❌ Empty file: {path}")
    print(f"✅ Loaded {os.path.basename(path)} → {df.shape}")
    return df

def normalize(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df

def prefix_grover(df, prefix):
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c.startswith("Grover_")})

# =============================================================================
# 2. PROTAC MAPPING & TENSOR CONSTRUCTION
# =============================================================================
def build_protac_tensor_dataset(protac_df, warhead_df, linker_df, e3_df):
    print("\n--- Mapping Components to PROTACs ---")
    rows = []
    for _, p in tqdm(protac_df.iterrows(), total=len(protac_df)):
        # Warhead (Uniprot preferred, fallback Target)
        wh = warhead_df[(warhead_df["warhead_Uniprot"] == p["warhead_Uniprot"]) |
                        (warhead_df["warhead_Target"]  == p["warhead_Target"])]
        if wh.empty: continue
        
        # Linker
        lk = linker_df[linker_df["linker_Compound_ID"] == p["Linker ID"]]
        if lk.empty: continue
        
        # E3
        e3 = e3_df[e3_df["E3_Ligase"] == p["E3 ligase"]]
        if e3.empty: continue
        
        rows.append({
            "PROTAC_Compound_ID": p["PROTAC_Compound_ID"],
            "warhead_Compound_ID": wh.iloc[0]["warhead_Compound_ID"],
            "linker_Compound_ID":  p["Linker ID"],
            "E3_Compound_ID":      e3.iloc[0]["E3_Compound_ID"]
        })

    protac_map = pd.DataFrame(rows)
    print(f"✅ Successfully Mapped PROTACs: {len(protac_map)}")

    print("--- Attaching Grover Features ---")
    protac_features = (
        protac_map
        .merge(warhead_df[["warhead_Compound_ID"] + [c for c in warhead_df.columns if c.startswith("W_Grover_")]], on="warhead_Compound_ID", how="inner")
        .merge(linker_df[["linker_Compound_ID"] + [c for c in linker_df.columns if c.startswith("L_Grover_")]], on="linker_Compound_ID", how="inner")
        .merge(e3_df[["E3_Compound_ID"] + [c for c in e3_df.columns if c.startswith("E_Grover_")]], on="E3_Compound_ID", how="inner")
    )

    w_cols = [c for c in protac_features.columns if c.startswith("W_Grover_")]
    l_cols = [c for c in protac_features.columns if c.startswith("L_Grover_")]
    e_cols = [c for c in protac_features.columns if c.startswith("E_Grover_")]
    assert len(w_cols) == len(l_cols) == len(e_cols), "❌ Grover dimensions mismatch across components!"
    
    # Build X Tensor Array
    X = np.stack([
        np.stack([
            r[w_cols].values.astype("float32"),
            r[l_cols].values.astype("float32"),
            r[e_cols].values.astype("float32"),
        ]) for _, r in protac_features.iterrows()
    ])
    
    print(f"🎯 Final X Tensor Shape: {X.shape}")
    return X, protac_features, len(w_cols)

# =============================================================================
# 3. PYTORCH MULTI-TASK ARCHITECTURE
# =============================================================================
class MultiTaskProtacModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        
        # Dual Prediction Heads
        self.head_dc50 = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.head_dmax = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        h = self.proj(x)                  
        h = self.transformer(h)           
        attn_scores = self.attn_pool(h)   
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (h * attn_weights).sum(dim=1) 
        
        return self.head_dc50(fused), self.head_dmax(fused), attn_weights

def masked_mse_loss(pred, target):
    """Safely calculates MSE ignoring NaN values in sparse PROTAC datasets."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])

# =============================================================================
# 4. MASTER EXECUTION & TRAINING
# =============================================================================
if __name__ == "__main__":
    # --- 1. Load Data ---
    BASE_DIR = "/storage/savi/saveenas/Projects/server_53/Projects/Synglue/Attention/data"
    protac_df  = load_full_data(f"{BASE_DIR}/Upadted_PROTAC_features.csv")
    warhead_df = load_full_data(f"{BASE_DIR}/Updated_warhead.csv")
    linker_df  = load_full_data(f"{BASE_DIR}/linker_updated_features.csv")
    e3_df      = load_full_data(f"{BASE_DIR}/E3_updated_features.csv")

    protac_df  = normalize(protac_df,  ["warhead_Uniprot", "warhead_Target", "E3 ligase"])
    warhead_df = normalize(warhead_df, ["warhead_Uniprot", "warhead_Target"])
    e3_df      = normalize(e3_df,      ["E3_Ligase"])

    warhead_df = prefix_grover(warhead_df, "W_")
    linker_df  = prefix_grover(linker_df,  "L_")
    e3_df      = prefix_grover(e3_df,      "E_")

    # --- 2. Build Tensors ---
    X, protac_features, feature_dim = build_protac_tensor_dataset(protac_df, warhead_df, linker_df, e3_df)

    # --- 3. Extract & Format Targets (DC50 & Dmax) ---
    print("\n--- Processing Biological Targets ---")
    label_df = protac_df[["PROTAC_Compound_ID", "DC50 (nM)", "Dmax (%)"]].copy()
    label_df["DC50 (nM)"] = pd.to_numeric(label_df["DC50 (nM)"], errors="coerce")
    label_df["Dmax (%)"] = pd.to_numeric(label_df["Dmax (%)"], errors="coerce")

    # Aggregate duplicates safely
    protac_labels = label_df.groupby("PROTAC_Compound_ID").agg({"DC50 (nM)": "median", "Dmax (%)": "max"}).reset_index()
    final_df = protac_features.merge(protac_labels, on="PROTAC_Compound_ID", how="left")

    # Format y_dc50 (Log10 transform valid numbers, keep NaN for missing)
    dc50_raw = final_df["DC50 (nM)"].values.astype(float)
    y_dc50 = np.where((dc50_raw > 0) & (~np.isnan(dc50_raw)), np.log10(dc50_raw), np.nan)
    
    # Format y_dmax
    y_dmax = final_df["Dmax (%)"].values.astype(float)

    print(f"Valid DC50 labels: {np.sum(~np.isnan(y_dc50))} | Valid Dmax labels: {np.sum(~np.isnan(y_dmax))}")

    # --- 4. Setup PyTorch Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Initializing Training on: {device}")
    
    X_tr, X_val, y_dc50_tr, y_dc50_val, y_dmax_tr, y_dmax_val = train_test_split(X, y_dc50, y_dmax, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_dc50_tr, dtype=torch.float32), torch.tensor(y_dmax_tr, dtype=torch.float32)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_dc50_val, dtype=torch.float32), torch.tensor(y_dmax_val, dtype=torch.float32)), batch_size=128)

    model = MultiTaskProtacModel(input_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- 5. Training Loop with Early Stopping ---
    EPOCHS = 100
    patience = 15
    best_loss = float("inf")
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb_dc50, yb_dmax in train_loader:
            xb, yb_dc50, yb_dmax = xb.to(device), yb_dc50.to(device), yb_dmax.to(device)
            optimizer.zero_grad()
            
            pred_dc50, pred_dmax, _ = model(xb)
            loss_dc50 = masked_mse_loss(pred_dc50.squeeze(), yb_dc50)
            loss_dmax = masked_mse_loss(pred_dmax.squeeze(), yb_dmax)
            
            # Dmax values (0-100) produce massive MSE compared to logDC50 (-2 to 4). We scale Dmax loss down by 0.01 to balance them.
            loss = loss_dc50 + (loss_dmax * 0.01)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb_dc50, yb_dmax in val_loader:
                xb, yb_dc50, yb_dmax = xb.to(device), yb_dc50.to(device), yb_dmax.to(device)
                pred_dc50, pred_dmax, _ = model(xb)
                val_loss += (masked_mse_loss(pred_dc50.squeeze(), yb_dc50) + (masked_mse_loss(pred_dmax.squeeze(), yb_dmax) * 0.01)).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), "best_multitask_transformer.pt")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    print("\n🎉 Training Complete! Model saved as 'best_multitask_transformer.pt'")