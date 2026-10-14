from pathlib import Path
import copy
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.patchtst import PatchTSTForecaster

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "germany_hourly_load.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "patchtst_metrics.csv"

LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 4
LEARNING_RATE = 0.001

PATCH_LEN = 24
STRIDE = 12
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
DROPOUT = 0.1

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_windows(values, lookback, horizon):
    X, y = [], []

    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback:i + lookback + horizon])

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.float32)

    return X, y


def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


def predict(model, loader, device):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            batch_predictions = model(X_batch)

            predictions.append(batch_predictions.cpu().numpy())
            actuals.append(y_batch.numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(actuals, axis=0),
    )


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return mae, rmse, wape


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    train_df = df[df["timestamp"] < "2019-01-01"].copy()
    val_df = df[
        (df["timestamp"] >= "2019-01-01")
        & (df["timestamp"] < "2020-01-01")
    ].copy()
    test_df = df[df["timestamp"] >= "2020-01-01"].copy()

    train_mean = train_df["demand_mw"].mean()
    train_std = train_df["demand_mw"].std()

    train_values = (train_df["demand_mw"].values - train_mean) / train_std
    val_values = (val_df["demand_mw"].values - train_mean) / train_std
    test_values = (test_df["demand_mw"].values - train_mean) / train_std

    X_train, y_train = make_windows(train_values, LOOKBACK, HORIZON)
    X_val, y_val = make_windows(val_values, LOOKBACK, HORIZON)
    X_test, y_test = make_windows(test_values, LOOKBACK, HORIZON)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, False)

    model = PatchTSTForecaster(
        lookback=LOOKBACK,
        horizon=HORIZON,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size

        train_loss = train_loss_sum / train_examples
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping.")
                break

    training_time = time.time() - start_time
    model.load_state_dict(best_state)

    val_pred_scaled, val_true_scaled = predict(model, val_loader, device)
    test_pred_scaled, test_true_scaled = predict(model, test_loader, device)

    val_pred = val_pred_scaled * train_std + train_mean
    val_true = val_true_scaled * train_std + train_mean
    test_pred = test_pred_scaled * train_std + train_mean
    test_true = test_true_scaled * train_std + train_mean

    val_mae, val_rmse, val_wape = calculate_metrics(val_true, val_pred)
    test_mae, test_rmse, test_wape = calculate_metrics(test_true, test_pred)

    print("\nValidation Results")
    print(f"MAE: {val_mae:.2f} MW | RMSE: {val_rmse:.2f} MW | WAPE: {val_wape:.4f}%")

    print("\nTest Results")
    print(f"MAE: {test_mae:.2f} MW | RMSE: {test_rmse:.2f} MW | WAPE: {test_wape:.4f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "model": "PatchTST-style",
        "val_mae_mw": val_mae,
        "val_rmse_mw": val_rmse,
        "val_wape_percent": val_wape,
        "test_mae_mw": test_mae,
        "test_rmse_mw": test_rmse,
        "test_wape_percent": test_wape,
        "training_time_seconds": training_time,
        "patch_len": PATCH_LEN,
        "stride": STRIDE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "num_layers": NUM_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
    }]).to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
