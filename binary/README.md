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
