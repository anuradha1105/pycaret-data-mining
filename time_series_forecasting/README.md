# Time Series Forecasting (Delhi Climate)

- Environment: Local (.venv-pycaret3, Python 3.11)
- Library: PyCaret 3.3.2
- Dataset: delhiclimate.csv
- Target variable: meantemp (daily mean temperature)
- Forecast horizon: 30 days
- Steps:
  1. Parsed date column and set as index
  2. Initialized PyCaret Time Series with seasonal_period=365
  3. Compared models using `compare_models()`
  4. Finalized best model and generated 30-day forecast
  5. Exported forecast to `forecast_output.csv`
