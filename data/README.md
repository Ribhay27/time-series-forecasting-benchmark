# Data

ForecastBench uses real-world hourly electricity-demand data for Germany from the **Open Power System Data (OPSD) Time Series dataset**, based on electricity-system data from the **ENTSO-E Transparency Platform**.

## Dataset

* **Region:** Germany
* **Frequency:** Hourly
* **Time period:** January 1, 2015 – September 30, 2020
* **Usable observations:** 50,400 hourly records
* **Target:** Actual electricity demand/load
* **Unit:** Megawatts (MW)

## Columns Used

The original OPSD dataset contains electricity-system variables for multiple European countries. ForecastBench initially uses only two columns:

| Original Column                      | Project Column | Description                                                |
| ------------------------------------ | -------------- | ---------------------------------------------------------- |
| `utc_timestamp`                      | `timestamp`    | Hour of the observation in UTC                             |
| `DE_load_actual_entsoe_transparency` | `demand_mw`    | Actual total electricity demand in Germany, measured in MW |

Each row represents one hour of electricity demand.

Example:

```text
timestamp                  demand_mw
2015-01-01 00:00:00+00:00  41151
2015-01-01 01:00:00+00:00  40135
2015-01-01 02:00:00+00:00  39106
```

## Prediction Target

`demand_mw` contains **historical actual electricity demand**.

ForecastBench uses past values of this variable to predict its future values.

The initial forecasting setup is:

```text
Previous 168 hours of demand
            ↓
          Model
            ↓
Next 24 hours of demand
```

Therefore:

* **Look-back window:** 168 hours
* **Forecast horizon:** 24 hours

These values will later be varied as part of the experimental study.

## Cleaning

The raw OPSD dataset was reduced to the Germany hourly-demand series.

Initial cleaning includes:

* selecting only the timestamp and German actual-load columns
* renaming columns to `timestamp` and `demand_mw`
* parsing timestamps as UTC datetimes
* converting demand values to numeric format
* removing rows without a valid demand observation
* sorting observations chronologically
* checking for duplicate timestamps
* checking for missing hourly timestamps

After cleaning:

* Missing demand values: **0**
* Duplicate timestamps: **0**
* Missing hourly timestamps: **0**

The resulting dataset contains **50,400 continuous hourly observations**.

## Data Storage

The full raw OPSD dataset is not stored directly in this repository.

```text
data/
├── README.md
├── raw/
└── processed/
```

* `raw/` — locally downloaded source data
* `processed/` — cleaned datasets produced by the preprocessing pipeline

The repository will contain code required to reproduce the cleaned dataset from the original source.

## Source

**Open Power System Data — Time Series, version 2020-10-06**

Underlying electricity-load data is sourced from the **ENTSO-E Transparency Platform**.
