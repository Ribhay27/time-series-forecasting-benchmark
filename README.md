# Electricity Demand Forecasting
## When Do Transformers Actually Win at Time-Series Forecasting?

A completed time-series forecasting study exploring whether additional model complexity delivers meaningful improvements in Germany's hourly electricity-demand forecasts. The project brings together Seasonal Naive, XGBoost, DLinear, LSTM and a PatchTST-style Transformer, with a focus on the relationship between forecast accuracy and model complexity.

Inspired by **Are Transformers Effective for Time Series Forecasting?** by **Zeng et al.** and **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** by **Nie et al.**, the study explores decomposition-based linear forecasting and patch-based attention alongside conventional baselines.

> **When is Transformer complexity actually worth it for time-series forecasting?**

## Project presentation

[View the Electricity Demand Forecasting Project Overview (PDF)](docs/Electricity_Demand_Forecasting_Overview.pdf)

A seven-slide walkthrough of the forecasting task, model choices, reported results and model-selection takeaways.

## Study at a glance

| Item | Description |
| --- | --- |
| Application | Germany electricity-demand forecasting |
| Dataset | Open Power System Data, Time Series version 2020-10-06 |
| Underlying source | ENTSO-E Transparency Platform |
| Documented coverage | January 2015–September 2020; 50,400 hourly observations |
| Target | Actual electricity demand in megawatts (MW) |
| Multi-step forecasting setup | Previous 168 hours → next 24 hours |
| Metrics | MAE, RMSE and WAPE |
| Tools | Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, Matplotlib and Jupyter |

## Project workflow

The study is organized around six stages:

1. **Research framing** — examine the case for simpler forecasting models and patch-based Transformers.
2. **Data preparation** — extract German electricity demand, parse UTC timestamps, clean observations and check hourly continuity.
3. **Exploration** — examine demand patterns, calendar profiles, rolling averages and autocorrelation.
4. **Model development** — implement a seasonal baseline, boosted trees, decomposition-based linear forecasting, recurrent modeling and patch-based attention.
5. **Evaluation** — calculate forecast errors using chronological evaluation and document model-specific settings.
6. **Communication** — summarize the findings and model-selection tradeoffs in the project presentation.

## Models

| Model | Approach |
| --- | --- |
| Seasonal Naive | Predict demand using the same hour from the previous week |
| XGBoost | Multi-output gradient-boosted trees using demand history, calendar features and historical means |
| DLinear | Moving-average decomposition with separate linear trend and remainder forecasts |
| LSTM | Recurrent sequence encoder with a multi-step forecast head |
| PatchTST-style | Historical patches, learned embeddings, self-attention and a forecast head |

## Results

### Reported study results

The [project presentation](docs/Electricity_Demand_Forecasting_Overview.pdf) reports the following WAPE values for the 2020 test period. Lower is better.

| Model | Reported WAPE |
| --- | ---: |
| PatchTST | 2.57% |
| XGBoost | 2.64% |
| DLinear | 3.04% |
| LSTM | 3.37% |
| Seasonal Naive | 4.20% |

In the presentation's reported results, PatchTST has the lowest WAPE, with approximately **0.07 percentage points** separating it from XGBoost. XGBoost and DLinear both have lower reported WAPE than LSTM.

**Complexity should earn its place.** The presentation's central takeaway is to consider the size of an accuracy gain alongside the effort involved in developing and maintaining a model.

### CSV result artifacts

The following values are recorded in [model_comparison_updated.csv](results/model_comparison_updated.csv) and [lstm_metrics_final.csv](results/lstm_metrics_final.csv), rounded for display.

| Model | MAE (MW) | RMSE (MW) | WAPE (%) | Artifact |
| --- | ---: | ---: | ---: | --- |
| Seasonal Naive | 2227.11 | 3781.28 | 4.1975 | Comparison CSV |
| XGBoost | 1399.16 | 2145.96 | 2.6370 | Comparison CSV |
| DLinear | 1610.52 | 2561.51 | 3.0354 | Comparison CSV |
| LSTM | 1789.39 | 2835.77 | 3.3725 | LSTM metrics CSV |

## Data preparation

See [data/README.md](data/README.md) for the dataset description.

The cleaning module reads `data/raw/time_series_60min_singleindex.csv` and selects:

| Source column | Output column |
| --- | --- |
| `utc_timestamp` | `timestamp` |
| `DE_load_actual_entsoe_transparency` | `demand_mw` |

It parses timestamps as UTC, converts demand to numeric values, drops invalid rows, sorts observations, keeps the first occurrence of duplicate timestamps and checks for missing hourly timestamps. The output is `data/processed/germany_hourly_load.csv`.

Raw and processed datasets are kept locally and excluded from Git.

## Repository structure

| Path | Contents |
| --- | --- |
| `data/README.md` | Dataset documentation |
| `docs/Electricity_Demand_Forecasting_Overview.pdf` | Project presentation |
| `notebooks/01_data_exploration.ipynb` | Demand exploration and visualization |
| `src/data/clean_data.py` | Data cleaning and validation |
| `src/baseline/seasonal_naive.py` | Weekly seasonal baseline |
| `src/models/XGBoost.py` | XGBoost features, training and evaluation |
| `src/models/dlinear.py` | DLinear model definition |
| `src/models/lstm.py` | LSTM model definition |
| `src/models/patchtst.py` | PatchTST-style model definition |
| `scripts/train_dlinear.py` | DLinear training-code excerpt |
| `scripts/train_lstm.py` | LSTM final-refit training |
| `scripts/train_patchtst_script.py` | PatchTST-style training |
| `scripts/tune_xgboost.py` | XGBoost validation search |
| `scripts/tune_patchtst.py` | PatchTST-style validation search |
| `results/model_comparison_updated.csv` | Seasonal Naive, XGBoost and DLinear metrics |
| `results/lstm_metrics_final.csv` | LSTM metrics and run settings |

## Usage

### 1. Environment

From a local checkout, create a Python environment and install the packages used by the source code:

```bash
git clone https://github.com/Ribhay27/time-series-forecasting-benchmark.git
cd time-series-forecasting-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn xgboost torch matplotlib jupyterlab
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

### 2. Data

Download `time_series_60min_singleindex.csv` from the OPSD Time Series **2020-10-06** release and place it in `data/raw/`.

```bash
mkdir -p data/raw
# Place the downloaded CSV in data/raw/.
python src/data/clean_data.py
```

The command prints the dataset's row count, date range and integrity checks, then saves the cleaned series.

### 3. Exploration

The notebook uses a path relative to `notebooks/`:

```bash
cd notebooks
jupyter lab 01_data_exploration.ipynb
# After stopping Jupyter, return to the repository root:
cd ..
```

### 4. Training and tuning entry points

Run these commands from the repository root after data preparation.

| Command | Output |
| --- | --- |
| `python src/baseline/seasonal_naive.py` | Validation and test metrics printed to the terminal |
| `python src/models/XGBoost.py` | Validation and test metrics for the configured 200-estimator model |
| `python scripts/tune_xgboost.py` | `results/xgboost_tuning.csv` |
| `python scripts/train_lstm.py` | `results/lstm_metrics.csv` |
| `python scripts/train_patchtst_script.py` | `results/patchtst_metrics.csv` |
| `python scripts/tune_patchtst.py` | `results/patchtst_tuning.csv` |

Training settings are defined in the scripts. Tuning outputs are separate from training configurations. Repeated runs replace files at the same output path; the committed comparison CSV and `lstm_metrics_final.csv` are retained study artifacts.

## Evaluation methodology

The source code uses chronological boundaries and the following model-specific procedures:

| Model | Evaluation procedure |
| --- | --- |
| Seasonal Naive | Weekly-lag predictions scored by hourly row; validation in 2019 and test from January 2020 |
| XGBoost | 168-hour history and 24-hour targets; training targets before 2019, validation targets within 2019 and test targets from January 2020 |
| LSTM | Final refit on pre-2020 data for 13 epochs; scaling fitted on that training period; 24-hour test targets from January 2020 |
| PatchTST-style | Training before 2019, validation in 2019 and test from January 2020; training-only scaling and windows constructed separately within each split |

XGBoost and LSTM allow historical context from before a target split boundary. PatchTST-style windows use the first 168 hours within each split as context. Multi-step metrics aggregate overlapping target windows; Seasonal Naive metrics aggregate individual hourly predictions. These details define how each script's metrics are calculated.

- **MAE:** mean absolute forecast error, in MW.
- **RMSE:** square root of mean squared forecast error, in MW.
- **WAPE:** `100 × sum(abs(actual − prediction)) / sum(abs(actual))`.
