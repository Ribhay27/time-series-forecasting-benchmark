from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/germany_hourly_load.csv")

LOOKBACK = 168
HORIZON = 24

TRAIN_END = pd.Timestamp("2019-01-01", tz="UTC")
VAL_END = pd.Timestamp("2020-01-01", tz="UTC")


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def create_windows(df):
    demand = df["demand_mw"].to_numpy()
    timestamps = df["timestamp"]

    X = []
    y = []
    forecast_start = []
    forecast_end = []

    for i in range(LOOKBACK, len(df) - HORIZON + 1):
        history = demand[i - LOOKBACK:i]
        target = demand[i:i + HORIZON]

        start_time = timestamps.iloc[i]
        end_time = timestamps.iloc[i + HORIZON - 1]

        calendar_features = np.array([
            start_time.hour,
            start_time.dayofweek,
            start_time.month,
            history[-24:].mean(),
            history.mean(),
        ])

        features = np.concatenate([history, calendar_features])

        X.append(features)
        y.append(target)
        forecast_start.append(start_time)
        forecast_end.append(end_time)

    return (
        np.array(X),
        np.array(y),
        pd.Series(forecast_start),
        pd.Series(forecast_end),
    )


def split_windows(X, y, forecast_start, forecast_end):
    train_mask = forecast_end < TRAIN_END

    val_mask = (
        (forecast_start >= TRAIN_END)
        & (forecast_end < VAL_END)
    )

    test_mask = forecast_start >= VAL_END

    X_train = X[train_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]

    X_val = X[val_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]

    X_test = X[test_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_forecast(y_true, y_pred):
    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    wape = np.sum(np.abs(error)) / np.sum(np.abs(y_true)) * 100

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

    X, y, forecast_start, forecast_end = create_windows(df)

    X_train, y_train, X_val, y_val, X_test, y_test = split_windows(
        X,
        y,
        forecast_start,
        forecast_end,
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    base_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model, n_jobs=1)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_results = evaluate_forecast(y_val, val_pred)
    test_results = evaluate_forecast(y_test, test_pred)

    print_results("Validation", val_results)
    print_results("Test", test_results)


if __name__ == "__main__":
    main()
