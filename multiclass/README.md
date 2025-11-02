
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
