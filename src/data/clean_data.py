"""Clean the OPSD hourly time-series dataset for ForecastBench.

This script extracts Germany's hourly electricity demand from the raw
Open Power System Data (OPSD) time-series CSV and saves a compact,
validated dataset for downstream forecasting experiments.

Expected input:
    data/raw/time_series_60min_singleindex.csv

Output:
    data/processed/germany_hourly_load.csv
"""

from pathlib import Path

import pandas as pd


RAW_FILE = Path("data/raw/time_series_60min_singleindex.csv")
OUTPUT_FILE = Path("data/processed/germany_hourly_load.csv")

TIMESTAMP_COLUMN = "utc_timestamp"
DEMAND_COLUMN = "DE_load_actual_entsoe_transparency"


def load_and_clean_data(input_path: Path) -> pd.DataFrame:
    """Load Germany hourly load data and apply basic cleaning."""

    df = pd.read_csv(
        input_path,
        usecols=[TIMESTAMP_COLUMN, DEMAND_COLUMN],
    )

    df = df.rename(
        columns={
            TIMESTAMP_COLUMN: "timestamp",
            DEMAND_COLUMN: "demand_mw",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["demand_mw"] = pd.to_numeric(
        df["demand_mw"],
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp", "demand_mw"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="first")

    return df.reset_index(drop=True)


def validate_data(df: pd.DataFrame) -> None:
    """Run basic integrity checks on the cleaned hourly series."""

    duplicate_timestamps = df["timestamp"].duplicated().sum()
    missing_values = df[["timestamp", "demand_mw"]].isna().sum().sum()

    expected_timestamps = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="h",
        tz="UTC",
    )

    missing_timestamps = expected_timestamps.difference(
        pd.DatetimeIndex(df["timestamp"])
    )

    print("\nDataset validation")
    print("------------------")
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['timestamp'].min()}")
    print(f"End: {df['timestamp'].max()}")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate timestamps: {duplicate_timestamps}")
    print(f"Missing hourly timestamps: {len(missing_timestamps)}")

    print("\nDemand summary (MW)")
    print("-------------------")
    print(df["demand_mw"].describe())

    if missing_values:
        raise ValueError("Cleaned dataset still contains missing values.")

    if duplicate_timestamps:
        raise ValueError("Cleaned dataset still contains duplicate timestamps.")

    if len(missing_timestamps):
        raise ValueError(
            f"Cleaned dataset is missing {len(missing_timestamps)} hourly timestamps."
        )


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}\n"
            "Place time_series_60min_singleindex.csv inside data/raw/ first."
        )

    df = load_and_clean_data(RAW_FILE)
    validate_data(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
