from pathlib import Path
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
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
FINAL_EPOCHS = 13
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "germany_hourly_load.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "lstm_metrics.csv"

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def create_windows(df):
    values = df["demand_scaled"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].tolist()
    splits = {"train": {"X": [], "y": []}, "test": {"X": [], "y": []}}

    for i in range(LOOKBACK, len(values) - HORIZON + 1):
        X = values[i - LOOKBACK:i]
        y = values[i:i + HORIZON]
        forecast_start = timestamps[i]
        forecast_end = timestamps[i + HORIZON - 1]

        if forecast_end < TEST_START:
            split = "train"
        elif forecast_start >= TEST_START:
            split = "test"
        else:
            continue

        splits[split]["X"].append(X)
        splits[split]["y"].append(y)

    return splits


def make_loader(X, y, shuffle=False, seed=42):
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def evaluate(y_true, y_pred):
    error = y_true - y_pred
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error ** 2))),
        "wape_percent": float(np.sum(np.abs(error)) / np.sum(np.abs(y_true)) * 100),
    }


def main():
    df = load_data(DATA_PATH)

    # Final refit: use 2015-2019 after tuning was completed on the 2019 validation period.
    final_train_mask = df["timestamp"] < TEST_START
    train_mean = df.loc[final_train_mask, "demand_mw"].mean()
    train_std = df.loc[final_train_mask, "demand_mw"].std()
    df["demand_scaled"] = (df["demand_mw"] - train_mean) / train_std

    splits = create_windows(df)
    train_loader = make_loader(splits["train"]["X"], splits["train"]["y"], shuffle=True)
    test_loader = make_loader(splits["test"]["X"], splits["test"]["y"])

    print(f"Final training examples: {len(train_loader.dataset):,}", flush=True)
    print(f"Test examples: {len(test_loader.dataset):,}", flush=True)

    sample_X, sample_y = next(iter(train_loader))
    print(f"X batch shape: {sample_X.shape}", flush=True)
    print(f"y batch shape: {sample_y.shape}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = LSTMForecaster(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        horizon=HORIZON,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.perf_counter()

    for epoch in range(FINAL_EPOCHS):
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
        print(f"Epoch {epoch + 1:02d}/{FINAL_EPOCHS} | Train Loss: {train_loss:.6f}", flush=True)

    training_seconds = time.perf_counter() - start_time

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

    print("\nFinal LSTM Test Results", flush=True)
    print(f"MAE:  {metrics['mae_mw']:.2f} MW", flush=True)
    print(f"RMSE: {metrics['rmse_mw']:.2f} MW", flush=True)
    print(f"WAPE: {metrics['wape_percent']:.4f}%", flush=True)
    print(f"Training time: {training_seconds:.2f} seconds", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "LSTM",
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "final_epochs": FINAL_EPOCHS,
        "trainable_parameters": parameter_count,
        "training_seconds": training_seconds,
        "MAE_MW": metrics["mae_mw"],
        "RMSE_MW": metrics["rmse_mw"],
        "WAPE_percent": metrics["wape_percent"],
    }]).to_csv(RESULTS_PATH, index=False)

    print(f"Saved metrics to: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
