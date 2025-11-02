# 🎯 Clustering – Customer Segmentation

This folder contains a minimal **PyCaret workflow** for this task.

---

## 📊 Dataset

- **File:** `Customer_Segmentation.csv`  
- **Features:** Relevant input attributes for modeling  
- **Target:** `Unlabeled Data`  
- **Purpose:** Group customers with similar purchasing behaviors using unsupervised learning.

---

## 🧠 Task

**Type:** Unsupervised Clustering  
**Goal:** Use PyCaret's AutoML to build, compare, and evaluate models for this task.

---

## 📓 Notebook

- **File:** `clustering.ipynb`  
- **Workflow:**
  1. Load and explore the dataset  
  2. Initialize `PyCaret` experiment setup  
  3. Compare multiple models automatically  
  4. Evaluate best-performing model visually  
  5. Generate and export predictions  
  6. Save the trained model  

---

## ⚙️ How to Run

Ensure PyCaret is installed:

```bash
pip install pycaret[analysis]
```

Run the notebook:

```bash
jupyter notebook clustering.ipynb
```

Then **Run All Cells** to:
- Load `Customer_Segmentation.csv`  
- Compare 15+ algorithms automatically  
- Select the best model based on chosen metric  
- Display evaluation plots and predictions  

---

## 💾 Model Output

- **Model File:** `best_model.pkl` *(excluded from Git)*  
- **Predictions:** Generated predictions with confidence/probability scores  

---

## 📈 Evaluation Metrics

- Accuracy / RMSE (depending on task)  
- Precision, Recall, F1-Score (for classification)  
- Confusion Matrix or Residual Plot  
- ROC-AUC or R2 Score  
- Model Comparison Table  

---

## 🎯 Use Case

Enables marketing teams to target customer segments with personalized campaigns.

---

## 🔗 References

- [PyCaret Official Tutorials](https://pycaret.gitbook.io/docs/get-started/tutorials)
