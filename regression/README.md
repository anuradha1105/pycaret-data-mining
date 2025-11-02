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
