from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/germany_hourly_load.csv")

SEASONAL_LAG = 168

TRAIN_END = "2019-01-01"
VAL_END = "2020-01-01"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def add_seasonal_naive_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = df["demand_mw"].shift(SEASONAL_LAG)
    return df


def split_data(df: pd.DataFrame):
    train = df[df["timestamp"] < TRAIN_END].copy()

    val = df[
        (df["timestamp"] >= TRAIN_END)
        & (df["timestamp"] < VAL_END)
    ].copy()

    test = df[df["timestamp"] >= VAL_END].copy()

    return train, val, test


def evaluate_forecast(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    wape = (
        np.sum(np.abs(y_true - y_pred))
        / np.sum(np.abs(y_true))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
    }


def print_results(name, results):
    print(f"\n{name}")
    print(f"MAE:  {results['MAE']:.2f} MW")
    print(f"RMSE: {results['RMSE']:.2f} MW")
    print(f"WAPE: {results['WAPE']:.2f}%")


def main():
    df = load_data(DATA_PATH)
    df = add_seasonal_naive_prediction(df)

    train, val, test = split_data(df)

    print("Train rows:", len(train))
    print("Validation rows:", len(val))
    print("Test rows:", len(test))

    val_results = evaluate_forecast(
        val["demand_mw"].to_numpy(),
        val["prediction"].to_numpy(),
    )

    test_results = evaluate_forecast(
        test["demand_mw"].to_numpy(),
        test["prediction"].to_numpy(),
    )

    print_results("Validation", val_results)
    print_results("Test", test_results)


if __name__ == "__main__":
    main()
