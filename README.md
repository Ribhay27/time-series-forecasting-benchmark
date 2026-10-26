
## When Do Transformers Actually Win at Time-Series Forecasting?

ForecastBench investigates whether Transformer-based forecasting provides enough predictive improvement to justify its complexity over simpler approaches, using Germany's hourly electricity demand.

The research motivation comes from **Are Transformers Effective for Time Series Forecasting?** (Zeng et al., DLinear) and **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** (Nie et al., PatchTST). This repository contains local model implementations, including a **PatchTST-style** patch-based Transformer; it does not establish a faithful reproduction of either paper's experiments.

> **Research question: When is Transformer complexity actually worth it for time-series forecasting?**

## Current status

Model code and partial result artifacts are checked in. The full benchmark is not yet reproducible end to end.

| Component | Verified repository status |
| --- | --- |
| Data preparation | Germany-load cleaning and integrity checks in `src/data/clean_data.py`; source CSV must be downloaded separately |
| Exploration | `notebooks/01_data_exploration.ipynb` contains data-quality, seasonality, distribution, rolling-average and autocorrelation analysis; cells have no saved execution outputs |
| Seasonal Naive | Weekly-lag baseline and validation/test evaluation code |
| XGBoost | Multi-output forecasting and a separate validation tuning script |
| DLinear | Model definition and a row in the comparison CSV; training script is incomplete and cannot run as checked in |
| LSTM | Model and final-refit training script; separate checked-in metrics CSV |
| PatchTST-style | Model, training and validation tuning scripts; no checked-in metrics CSV |

### Recorded results

Values below are rounded from [results/model_comparison_updated.csv](results/model_comparison_updated.csv).

| Model | MAE (MW) | RMSE (MW) | WAPE (%) |
| --- | ---: | ---: | ---: |
| Seasonal Naive | 2227.11 | 3781.28 | 4.1975 |
| XGBoost | 1399.16 | 2145.96 | 2.6370 |
| DLinear | 1610.52 | 2561.51 | 3.0354 |

The separate [results/lstm_metrics_final.csv](results/lstm_metrics_final.csv) records LSTM MAE **1789.39 MW**, RMSE **2835.77 MW**, and WAPE **3.3725%**, with a 168-hour lookback, 24-hour horizon and 13 training epochs. It is not included in the comparison CSV.

These are **checked-in values, not independently reproduced results**. The comparison CSV does not record configurations, split boundaries, forecast origins or run identifiers. XGBoost has the lowest recorded errors among its three rows, but this is not sufficient to establish a controlled ranking across all five models. No PatchTST result or Transformer-win claim is supported by the checked-in result files.

## Data

The documented source is **Open Power System Data (OPSD), Time Series version 2020-10-06**, using Germany load data from the ENTSO-E Transparency Platform. See [data/README.md](data/README.md).

- Input: `data/raw/time_series_60min_singleindex.csv`.
- Required columns: `utc_timestamp` and `DE_load_actual_entsoe_transparency`.
- Output: `data/processed/germany_hourly_load.csv`, with UTC `timestamp` and `demand_mw` (MW).
- The data README reports 50,400 hourly observations from January 1, 2015 through September 30, 2020. Neither raw nor processed data is checked in; verify the actual row count and date range from the cleaner's output.

The cleaner parses timestamps and demand, drops invalid rows, sorts chronologically, removes duplicate timestamps (keeping the first), and rejects remaining missing hourly timestamps. It does **not** download data, fill gaps or enforce the documented date range. EIA and NOAA inputs are not part of the checked-in pipeline.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/README.md` | Dataset documentation |
| `notebooks/01_data_exploration.ipynb` | Exploration of the locally cleaned series |
| `src/data/clean_data.py` | Clean and validate the raw hourly CSV |
| `src/baseline/seasonal_naive.py` | 168-hour seasonal-lag baseline |
| `src/models/XGBoost.py` | XGBoost window creation, training and evaluation |
| `src/models/dlinear.py` | Moving-average decomposition and two linear forecast heads |
| `src/models/lstm.py` | LSTM forecast model |
| `src/models/patchtst.py` | PatchTST-style forecast model |
| `scripts/train_dlinear.py` | Incomplete DLinear training fragment |
| `scripts/train_lstm.py` | LSTM final refit and test evaluation |
| `scripts/train_patchtst_script.py` | PatchTST-style training with early stopping |
| `scripts/tune_xgboost.py` | Three-configuration validation search |
| `scripts/tune_patchtst.py` | Five-configuration validation search |
| `results/model_comparison_updated.csv` | Recorded Seasonal Naive, XGBoost and DLinear metrics |
| `results/lstm_metrics_final.csv` | Recorded LSTM metrics and run settings |

Raw/processed data directories are created locally and their contents are ignored by Git. Model checkpoint extensions are also ignored. There is no checked-in dependency manifest, lockfile, shared benchmark runner or results-aggregation script.

## Run flow

The commands below follow the checked-in paths and entry points. They have not been validated by a full training rerun, and they do not guarantee recreation of the recorded CSV values.

### 1. Set up an environment

Run from the repository root. This is a minimal dependency list inferred from imports, not a pinned or tested environment specification.

```bash
git clone https://github.com/Ribhay27/time-series-forecasting-benchmark.git
cd time-series-forecasting-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn xgboost torch matplotlib jupyterlab
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Record Python/package versions and hardware with any new experiment; fixed seeds alone do not guarantee identical results across environments.

### 2. Obtain and clean the data

Download `time_series_60min_singleindex.csv` from the OPSD Time Series **2020-10-06** release, create `data/raw/`, and place the file there. There is no ingestion/download script in this repository.

```bash
mkdir -p data/raw
# Place the downloaded CSV in data/raw/ before continuing.
python src/data/clean_data.py
```

The cleaner prints row counts, date bounds and integrity checks, then writes `data/processed/germany_hourly_load.csv`. Resolve any reported gaps before running models.

For optional exploration, launch Jupyter from `notebooks/` so the notebook's `../data/processed/` path resolves:

```bash
cd notebooks
jupyter lab 01_data_exploration.ipynb
# After stopping Jupyter:
cd ..
```

### 3. Run available model entry points

These commands are independent after cleaning; tuning does not automatically update training-script settings.

| Command (from repository root) | Behavior/output |
| --- | --- |
| `python src/baseline/seasonal_naive.py` | Prints validation/test MAE, RMSE and WAPE; no CSV export |
| `python src/models/XGBoost.py` | Fits the hard-coded 200-estimator model and prints validation/test metrics; no CSV export |
| `python scripts/tune_xgboost.py` | Evaluates validation configurations and writes `results/xgboost_tuning.csv`; does not perform final test evaluation |
| `python scripts/train_lstm.py` | Refits on pre-2020 data for 13 epochs and writes `results/lstm_metrics.csv` |
| `python scripts/train_patchtst_script.py` | Uses validation early stopping, evaluates validation/test sets and writes `results/patchtst_metrics.csv` |
| `python scripts/tune_patchtst.py` | Searches validation configurations and writes `results/patchtst_tuning.csv` |

Generated files overwrite the same output path on subsequent runs. Note that the LSTM script writes `lstm_metrics.csv`, whereas the committed artifact is named `lstm_metrics_final.csv`. None of these commands rebuilds `model_comparison_updated.csv`.

**DLinear blocker:** `scripts/train_dlinear.py` begins with an indented statement at line 1 and fails Python parsing with `IndentationError: unexpected indent`. Imports, configuration, data/loader setup and the `main()` definition are absent. The model definition alone is not an executable training pipeline; restore the missing trainer before attempting to reproduce its recorded result.

## Evaluation details and limits

- **Seasonal Naive:** predicts each timestamp from its observed demand 168 hours earlier. It evaluates individual hourly rows, with validation in 2019 and test from January 1, 2020.
- **XGBoost:** uses 168 historical demand values plus forecast-start hour, weekday, month and 24/168-hour historical means to predict the next 24 hours. Target windows must lie wholly within train (before 2019), validation (2019) or test (2020 onward); historical context may cross a split boundary.
- **LSTM:** the committed script is a final refit on all pre-2020 targets, including 2019. Normalization uses pre-2020 demand. It predicts 24 hours from 168 hours of history, allows pre-test context and excludes targets that straddle the test boundary. A comment refers to earlier validation tuning, but no LSTM tuning script or tuning results are checked in.
- **PatchTST-style:** trains before 2019, validates on 2019 and tests on 2020 onward. Scaling uses training data only. Windows are built separately inside each split, so the first 168 hours of each split are context rather than forecast targets.
- **DLinear:** its complete split, scaling and window-generation procedure cannot be verified from the incomplete trainer.

MAE and RMSE are in MW; WAPE is `100 * sum(abs(actual - prediction)) / sum(abs(actual))`. The multi-step scripts aggregate over overlapping 24-hour target windows, while the baseline scores each hourly row once. Differences in target coverage, weighting and training periods must be reconciled before claiming an apples-to-apples comparison. Chronological splits and training-only scaling in the inspected scripts are useful safeguards, not proof of a fully audited leakage-free benchmark.

## Remaining work

- Restore the DLinear trainer and add a pinned environment with a verified clean-checkout run.
- Standardize forecast origins, target coverage, training/refit policy and metric aggregation across models.
- Preserve configurations, seeds, environment details and result provenance; automate comparison CSV generation.
- Commit reproducible PatchTST-style results and integrate LSTM only after aligning evaluation protocols.
- Extend experiments across horizons, context lengths, training-data sizes and seasonal/unusual demand periods.
- Measure comparable training/inference costs and perform repeated-run, statistical and error analyses.

The research question remains open. The repository currently supports inspecting model implementations and partial recorded results; it does not yet establish when Transformer complexity earns its place.
