# ForecastBench

### When Do Transformers Actually Win at Time-Series Forecasting?

**ForecastBench** is a research reproduction and empirical benchmarking project investigating when Transformer-based forecasting models provide enough predictive improvement to justify their additional complexity over simpler approaches.

The project is motivated by two influential time-series forecasting papers:

* **Are Transformers Effective for Time Series Forecasting?** — Zeng et al.
  Introduced the LTSF-Linear family, including **DLinear**, and demonstrated that simple linear models can outperform several Transformer-based forecasting architectures.

* **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** — Nie et al.
  Introduced **PatchTST**, which uses patch-based representations and channel independence to improve Transformer forecasting performance.

## Research Question

> **When is Transformer complexity actually worth it for time-series forecasting?**

Rather than simply identifying the model with the lowest error, ForecastBench will evaluate models across:

* Forecast accuracy
* Forecast horizon
* Historical context length
* Training-data availability
* Seasonal and unusual demand periods
* Computational cost
* Robustness
* Model complexity

## Planned Models

The project will compare a progression of forecasting approaches:

1. Seasonal Naive
2. XGBoost
3. DLinear
4. LSTM
5. PatchTST

This creates a deliberate complexity ladder from simple statistical baselines to modern Transformer-based forecasting.

## Real-World Application

The primary application will be **hourly electricity-demand forecasting** using authoritative historical electricity data.

The project will use chronological evaluation and leakage-safe preprocessing to reflect realistic forecasting conditions.

## Project Goals

ForecastBench will:

* Reproduce key aspects of DLinear and PatchTST research
* Benchmark simple and complex forecasting architectures under controlled conditions
* Study how model rankings change across forecast horizons and look-back windows
* Evaluate robustness across different operating conditions
* Compare predictive gains against training and inference costs
* Perform statistical and error analysis rather than relying only on aggregate metrics
* Determine when additional architectural complexity provides practically meaningful value

## Project Status

🚧 **In development**

Current stage:

**Research verification and project foundation**

Experimental results will only be reported after the models and evaluation pipeline have been implemented and validated.

## Planned Stack

* Python
* PyTorch
* pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* EIA electricity data
* NOAA weather data where appropriate

Additional engineering tools will be introduced only where they meaningfully support reproducibility, experimentation, or deployment.


## Source:
Open Power System Data — Time Series
Version: 2020-10-06
Primary source: ENTSO-E Transparency Platform

The raw dataset is not stored in this repository.

Run the data ingestion script to download and extract
Germany's hourly electricity demand series.

## Research Principles

This project follows several rules:

* No fabricated results or metrics
* No random train/test splitting for time-series data
* No future-information leakage
* Strong simple baselines before complex models
* Reproducible experiment configurations
* Clear separation between paper reproduction and original extensions
* No assumption that the most complex model will win

Possible outcomes include DLinear winning, PatchTST winning, conventional ML winning, or different models being preferable under different conditions.

That outcome will be determined by the experiments.

---

**ForecastBench is currently under active development.**
