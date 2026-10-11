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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.lstm import LSTMForecaster


LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 2
HIDDEN_SIZE = 32
NUM_LAYERS = 1
CPU_THREADS = 1

VAL_START = pd.Timestamp("2019-01-01", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "germany_hourly_load.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "lstm_metrics.csv"


np.random.seed(42)
torch.manual_seed(42)

if not torch.cuda.is_available():
    torch.set_num_threads(CPU_THREADS)


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def create_windows(df):
    values = df["demand_scaled"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].tolist()

    splits = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []},
    }

    for i in range(LOOKBACK, len(values) - HORIZON + 1):
        X = values[i - LOOKBACK:i]
        y = values[i:i + HORIZON]

        forecast_start = timestamps[i]
        forecast_end = timestamps[i + HORIZON - 1]

        if forecast_end < VAL_START:
            split = "train"
        elif forecast_start >= VAL_START and forecast_end < TEST_START:
            split = "val"
        elif forecast_start >= TEST_START:
            split = "test"
        else:
            continue

        splits[split]["X"].append(X)
        splits[split]["y"].append(y)

    return splits


def make_loader(X, y, shuffle=False):
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.float32)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def evaluate(y_true, y_pred):
    error = y_true - y_pred

    return {
        "mae_mw": np.mean(np.abs(error)),
        "rmse_mw": np.sqrt(np.mean(error ** 2)),
        "wape_percent": (
            np.sum(np.abs(error))
            / np.sum(np.abs(y_true))
            * 100
        ),
    }


def main():
    df = load_data(DATA_PATH)

    train_mask = df["timestamp"] < VAL_START
    train_mean = df.loc[train_mask, "demand_mw"].mean()
    train_std = df.loc[train_mask, "demand_mw"].std()

    df["demand_scaled"] = (
        df["demand_mw"] - train_mean
    ) / train_std

    splits = create_windows(df)

    train_loader = make_loader(
        splits["train"]["X"],
        splits["train"]["y"],
        shuffle=True,
    )
    val_loader = make_loader(
        splits["val"]["X"],
        splits["val"]["y"],
    )
    test_loader = make_loader(
        splits["test"]["X"],
        splits["test"]["y"],
    )

    print(f"Training examples: {len(train_loader.dataset):,}")
    print(f"Validation examples: {len(val_loader.dataset):,}")
    print(f"Test examples: {len(test_loader.dataset):,}")

    sample_X, sample_y = next(iter(train_loader))
    print(f"X batch shape: {sample_X.shape}")
    print(f"y batch shape: {sample_y.shape}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = LSTMForecaster(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        horizon=HORIZON,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    training_seconds = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("No model state was saved.")

    model.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    predictions_mw = predictions * train_std + train_mean
    actuals_mw = actuals * train_std + train_mean

    metrics = evaluate(actuals_mw, predictions_mw)

    print("\nLSTM Test Results")
    print(f"MAE:  {metrics['mae_mw']:.2f} MW")
    print(f"RMSE: {metrics['rmse_mw']:.2f} MW")
    print(f"WAPE: {metrics['wape_percent']:.4f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame([
        {
            "model": "LSTM",
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "best_epoch": best_epoch,
            "trainable_parameters": parameter_count,
            "training_seconds": training_seconds,
            "MAE_MW": metrics["mae_mw"],
            "RMSE_MW": metrics["rmse_mw"],
            "WAPE_percent": metrics["wape_percent"],
        }
    ])

    result.to_csv(RESULTS_PATH, index=False)
    print(f"Saved metrics to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
