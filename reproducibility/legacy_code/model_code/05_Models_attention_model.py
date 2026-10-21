import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "05_Models/"
MODEL_PATH = "05_Models/best_attention_model.pt"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
PATIENCE = 10
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_dc50.npy"))

    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    return X, y

# =========================================================
# SPLIT DATA
# =========================================================

def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# =========================================================
# DATALOADERS
# =========================================================

def build_loaders(X_train, X_val, X_test, y_train, y_val, y_test):

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    return train_loader, val_loader, test_loader

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_model(model, train_loader, val_loader):

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    counter = 0

    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            pred, _ = model(xb)
            loss = loss_fn(pred.squeeze(), yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                pred, _ = model(xb)
                loss = loss_fn(pred.squeeze(), yb)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered")
                break

# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, test_loader):

    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)

            pred, _ = model(xb)

            preds.extend(pred.squeeze().cpu().numpy())
            trues.extend(yb.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    print("\n=== Test Performance ===")
    print("R²   :", r2_score(trues, preds))
    print("RMSE :", mean_squared_error(trues, preds, squared=False))
    print("MAE  :", mean_absolute_error(trues, preds))

# =========================================================
# MAIN
# =========================================================

def main():

    from model_architecture import ProtacAttentionModel  # keep model separate

    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    train_loader, val_loader, test_loader = build_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    model = ProtacAttentionModel(input_dim=X.shape[-1]).to(DEVICE)

    train_model(model, train_loader, val_loader)

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))

    evaluate_model(model, test_loader)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()