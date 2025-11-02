# 🧠 PyCaret Machine Learning Assignment 

This repository contains a complete collection of **PyCaret-based machine learning workflows** implemented as part of the *Data Science and Machine Learning* coursework.  
The project covers all major ML problem types — from classification and regression to clustering, anomaly detection, and time series forecasting — all executed using **PyCaret’s low-code AutoML framework**.

---

## 📚 Project Overview

Each notebook demonstrates how to rapidly build, compare, and deploy models using PyCaret with minimal code.  
The repository is organized into modular folders — one for each machine learning level.

| **Category** | **Description** | **Example Dataset** |
|---------------|-----------------|----------------------|
| **Binary Classification** | Predict yes/no outcomes | Telco Customer Churn |
| **Multiclass Classification** | Predict multiple categories | Iris Flower Dataset |
| **Regression** | Predict continuous values | Boston Housing |
| **Clustering** | Group unlabeled data | Customer Segmentation |
| **Anomaly Detection** | Detect outliers | Credit Card Fraud |
| **Time Series Forecasting** | Predict temporal trends | Delhi Climate Data |

---

## 🧩 CRISP–DM Framework Alignment

This assignment is structured following the **CRISP–DM** methodology:

1. **Business Understanding** – Define ML objective and target variable.  
2. **Data Understanding** – Explore and visualize data for key patterns.  
3. **Data Preparation** – Handle missing values, normalization, and feature engineering.  
4. **Modeling** – Apply PyCaret AutoML to train and compare models.  
5. **Evaluation** – Select best model based on accuracy, AUC, or RMSE.  
6. **Deployment** – Save and reuse finalized model for inference.

---

## 🧠 Technologies & Tools Used

- **PyCaret 3.3.x** – Low-code ML framework  
- **Python 3.11** (macOS / Colab)  
- **pandas**, **numpy**, **matplotlib**, **seaborn**  
- **VS Code / Google Colab** for development  
- **GitHub** for version control and submission  
- **Video walkthroughs** for each notebook execution  

---

**Folder Structure**
pycaret_assignment/
├── anomaly_detection/          # Fraud detection on bank transactions
├── association_rule_mining/    # Market basket analysis on grocery data
├── clustering/                 # Customer segmentation
├── multiclass/                 # Student performance classification
├── regression/                 # Car price prediction
├── time_series/                # Weather forecasting (univariate)
└── README.md                   # This file

Each folder contains:
📓 Jupyter Notebook (.ipynb) with complete workflow
📊 Dataset (.csv)
📝 Detailed README with task description and instructions
🔧 Saved models (.pkl - excluded from Git)


## 📹 Deliverables

- Google Colab or VS Code notebooks for each ML level  
- 1-minute **video walkthroughs** explaining each notebook  
- Organized **GitHub directory** with data, code, and logs  
- Consistent **PyCaret environment setup** (venv + Colab)


---

## 🧩 1. Binary Classification – Level
**Dataset:** `telco_churn.csv`  
**Objective:** Predict whether a telecom customer will churn (Yes/No).  

### Steps:
1. Load and clean the dataset (handle missing values, numeric conversions).
2. Use **PyCaret Classification Module** with normalization and multicollinearity removal.
3. Compare models and finalize the best one.

```python
from pycaret.classification import *
df = pd.read_csv("data/telco_churn.csv")
s = setup(data=df, target="Churn", normalize=True, session_id=42)
best = compare_models()
final_model = finalize_model(best)
save_model(final_model, "telco_churn_pycaret")
```

**Result:** LightGBM achieved the highest AUC (~0.83).

---

## 🌸 2. Multiclass Classification – Level
**Dataset:** `iris.csv`  
**Objective:** Predict the species of iris flowers.  

### Steps:
- Load the Iris dataset and setup PyCaret for multiclass classification.
- Compare models and finalize the best one.

```python
from pycaret.classification import *
df = pd.read_csv("data/iris.csv")
s = setup(data=df, target="species", session_id=42)
best = compare_models()
final_model = finalize_model(best)
save_model(final_model, "iris_multiclass_pycaret")
```

**Result:** Random Forest achieved >95% accuracy.

---

## 🏠 3. Regression – Level
**Dataset:** `boston_housing.csv`  
**Objective:** Predict median house price (`MEDV`).  

```python
from pycaret.regression import *
df = pd.read_csv("data/boston_housing.csv")
s = setup(data=df, target="MEDV", session_id=42)
best = compare_models()
final_model = finalize_model(best)
save_model(final_model, "boston_regression_pycaret")
```

**Result:** LightGBM achieved R² ≈ 0.91.

---

## 🎯 4. Clustering – Level
**Dataset:** `segmentation_data.csv`  
**Objective:** Group customers into meaningful clusters.

```python
from pycaret.clustering import *
df = pd.read_csv("data/segmentation_data.csv")
s = setup(data=df, session_id=42, normalize=True)
kmeans = create_model('kmeans')
plot_model(kmeans, plot='cluster')
```

**Result:** 4 clusters identified with distinct behavior patterns.

---

## 🚨 5. Anomaly Detection – Level
**Dataset:** `credit_card_fraud.csv`  
**Objective:** Detect fraudulent transactions.  

```python
from pycaret.anomaly import *
df = pd.read_csv("data/credit_card_fraud.csv")
s = setup(data=df, session_id=42)
iforest = create_model('iforest')
preds = assign_model(iforest)
plot_model(iforest, plot='tsne')
```

**Result:** Isolation Forest effectively flagged anomalies (~1.8%).

---

## ⏳ 6. Time Series Forecasting

### 📈 a) Univariate Forecasting (No Exogenous Variables)
**Dataset:** `delhiclimate.csv`  
**Objective:** Predict daily mean temperature for the next 30 days.

```python
from pycaret.time_series import *
df = pd.read_csv("data/delhiclimate.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
y = df['meantemp']

exp = setup(data=y, fh=30, fold=3, session_id=42, seasonal_period=365)
best = compare_models()
final_model = finalize_model(best)
forecast = predict_model(final_model, fh=30)
```

**Result:** Accurate 30-day forecast following historical seasonal trends.

---

### 🌦️ b) Univariate Forecasting (With Exogenous Variables)
**Dataset:** `delhiclimate.csv`  
**Objective:** Forecast temperature with humidity, wind speed, and pressure as exogenous variables.

```python
from pycaret.time_series import *
y = df['meantemp']
X = df[['humidity', 'wind_speed', 'meanpressure']]

exp = setup(data=y, fh=30, fold=3, session_id=42, seasonal_period=365)
best = compare_models(sort="MASE")
final_model = finalize_model(best)

# Build future exogenous data
future_index = pd.date_range(start=df.index.max()+pd.Timedelta(days=1), periods=30, freq='D')
future_exog = pd.concat([X.iloc[[-1]]]*30, ignore_index=True)
future_exog.index = future_index.to_period('D')

forecast = predict_model(final_model, fh=30, X=future_exog)
forecast.to_csv("forecast_with_exog_output.csv", index=False)
```

**Result:** 30-day forecast incorporating external weather variables.

