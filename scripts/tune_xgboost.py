from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/germany_hourly_load.csv")
RESULTS_PATH = Path("results/xgboost_tuning.csv")

LOOKBACK = 168
HORIZON = 24

TRAIN_END = pd.Timestamp("2019-01-01", tz="UTC")
VAL_END = pd.Timestamp("2020-01-01", tz="UTC")


PARAM_GRID = [
    {"n_estimators": 75, "max_depth": 3, "learning_rate": 0.08},
    {"n_estimators": 125, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def create_windows(df):
    demand = df["demand_mw"].to_numpy()

    X = []
    y = []
    forecast_start = []
    forecast_end = []

    for i in range(LOOKBACK, len(df) - HORIZON + 1):
        history = demand[i - LOOKBACK:i]
        target = demand[i:i + HORIZON]

        start_time = df["timestamp"].iloc[i]
        end_time = df["timestamp"].iloc[i + HORIZON - 1]

        extra_features = np.array([
            start_time.hour,
            start_time.dayofweek,
            start_time.month,
            history[-24:].mean(),
            history.mean(),
        ])

        X.append(np.concatenate([history, extra_features]))
        y.append(target)
        forecast_start.append(start_time)
        forecast_end.append(end_time)

    return (
        np.array(X),
        np.array(y),
        pd.Series(forecast_start),
        pd.Series(forecast_end),
    )


def split_train_validation(X, y, forecast_start, forecast_end):
    train_mask = forecast_end < TRAIN_END

    val_mask = (
        (forecast_start >= TRAIN_END)
        & (forecast_end < VAL_END)
    )

    return (
        X[train_mask.to_numpy()],
        y[train_mask.to_numpy()],
        X[val_mask.to_numpy()],
        y[val_mask.to_numpy()],
    )


def evaluate(y_true, y_pred):
    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    wape = np.sum(np.abs(error)) / np.sum(np.abs(y_true)) * 100

    return mae, rmse, wape


def main():
    df = load_data(DATA_PATH)

    X, y, forecast_start, forecast_end = create_windows(df)

    X_train, y_train, X_val, y_val = split_train_validation(
        X,
        y,
        forecast_start,
        forecast_end,
    )

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    results = []

    for params in PARAM_GRID:
        print("\nTesting:", params)

        base_model = XGBRegressor(
            **params,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

        model = MultiOutputRegressor(base_model, n_jobs=1)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        y_pred = model.predict(X_val)

        mae, rmse, wape = evaluate(y_val, y_pred)

        results.append({
            **params,
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape,
            "train_seconds": train_seconds,
        })

        print(f"MAE:  {mae:.2f} MW")
        print(f"RMSE: {rmse:.2f} MW")
        print(f"WAPE: {wape:.2f}%")
        print(f"Train time: {train_seconds:.1f} s")

    results_df = pd.DataFrame(results).sort_values("WAPE")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nValidation ranking")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]

    print("\nBest validation configuration")
    print(
        f"n_estimators={int(best['n_estimators'])}, "
        f"max_depth={int(best['max_depth'])}, "
        f"learning_rate={best['learning_rate']}"
    )
    print(f"Validation WAPE: {best['WAPE']:.2f}%")


if __name__ == "__main__":
    main()
