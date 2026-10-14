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
RESULTS_PATH = PROJECT_ROOT / "results" / "patchtst_tuning.csv"

LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3
SEED = 42

PARAM_GRID = [
    {
        "patch_len": 24,
        "stride": 12,
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "patch_len": 24,
        "stride": 12,
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "learning_rate": 0.0005,
    },
    {
        "patch_len": 24,
        "stride": 12,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "patch_len": 12,
        "stride": 6,
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
    {
        "patch_len": 24,
        "stride": 12,
        "d_model": 64,
        "n_heads": 4,
        "num_layers": 3,
        "d_ff": 128,
        "dropout": 0.1,
        "learning_rate": 0.001,
    },
]


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
    print(f"Using device: {device}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    train_df = df[df["timestamp"] < "2019-01-01"].copy()
    val_df = df[
        (df["timestamp"] >= "2019-01-01")
        & (df["timestamp"] < "2020-01-01")
    ].copy()

    train_mean = train_df["demand_mw"].mean()
    train_std = train_df["demand_mw"].std()

    train_values = (train_df["demand_mw"].values - train_mean) / train_std
    val_values = (val_df["demand_mw"].values - train_mean) / train_std

    X_train, y_train = make_windows(train_values, LOOKBACK, HORIZON)
    X_val, y_val = make_windows(val_values, LOOKBACK, HORIZON)

    val_loader = make_loader(X_val, y_val, BATCH_SIZE, False)

    results = []

    for config_number, params in enumerate(PARAM_GRID, start=1):
        print("\n" + "=" * 60)
        print(f"Configuration {config_number}/{len(PARAM_GRID)}")
        print(params)
        print("=" * 60)

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_loader = make_loader(X_train, y_train, BATCH_SIZE, True)

        model = PatchTSTForecaster(
            lookback=LOOKBACK,
            horizon=HORIZON,
            patch_len=params["patch_len"],
            stride=params["stride"],
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            num_layers=params["num_layers"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
        )

        best_val_loss = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0

        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

            val_loss = evaluate_loss(
                model,
                val_loader,
                criterion,
                device,
            )

            print(f"Epoch {epoch:02d} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print("Early stopping.")
                    break

        training_time = time.time() - start_time
        model.load_state_dict(best_state)

        val_pred_scaled, val_true_scaled = predict(
            model,
            val_loader,
            device,
        )

        val_pred = val_pred_scaled * train_std + train_mean
        val_true = val_true_scaled * train_std + train_mean

        val_mae, val_rmse, val_wape = calculate_metrics(
            val_true,
            val_pred,
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        results.append({
            "config_number": config_number,
            **params,
            "best_epoch": best_epoch,
            "val_mae_mw": val_mae,
            "val_rmse_mw": val_rmse,
            "val_wape_percent": val_wape,
            "trainable_params": trainable_params,
            "training_time_seconds": training_time,
        })

        print(f"Validation WAPE: {val_wape:.4f}%")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_wape_percent").reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nPATCHTST TUNING RESULTS")
    print(
        results_df[
            [
                "config_number",
                "patch_len",
                "stride",
                "d_model",
                "num_layers",
                "learning_rate",
                "best_epoch",
                "val_wape_percent",
            ]
        ].to_string(index=False)
    )

    print("\nBest configuration:")
    print(results_df.iloc[0].to_dict())

    print(f"\nSaved tuning results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
