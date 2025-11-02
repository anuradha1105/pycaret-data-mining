PyCaret Assignment – Anuradha Srivastav
Overview
This repository contains the end-to-end implementation of all PyCaret modules for low-code machine learning tasks. Each task was completed in a Google Colab environment and verified in a local virtual environment through VS Code.
Structure
Each folder represents a distinct PyCaret functionality:- classification/: Binary and Multiclass classification notebooks.- regression/: Regression analysis notebooks.- clustering/: Unsupervised clustering demonstration.- anomaly_detection/: Anomaly detection implementation.- association_rule_mining/: Association Rules with legacy PyCaret 2.3.5.- time_series/: Time Series forecasting (univariate & with exogenous variables).- gradio/: Interactive web app demos using Gradio.
Virtual Environments
1. `.venv-pycaret3` – Python 3.11 with PyCaret 3.x for all tasks except Association Rules.2. `.venv-pycaret2` – Python 3.8.18 with PyCaret 2.3.5 exclusively for Association Rules.
Execution Instructions
To reproduce the results:```bashgit clone https://github.com/anuradhasrivastav/pycaret-assignment.gitcd pycaret-assignmentsource .venv-pycaret3/bin/activate  # or .venv-pycaret2 for association rulesjupyter notebook```Open the desired notebook (e.g., `association_rule_mining/pycaret_association_rules.ipynb`) and run all cells.
Recording Instructions
Each notebook is demonstrated in a 1-minute video walkthrough showing:1. Import statements and setup function.2. Dataset loading and exploration.3. PyCaret setup and model creation.4. Output demonstration and model save.Videos are stored under `videos/` with the naming convention `01-classification.mp4`, `02-regression.mp4`, etc.
Gradio Demos
Two Gradio apps demonstrate PyCaret integration for deployment:- `gradio/gradio_classification_app.ipynb`- `gradio/gradio_regression_app.ipynb`Run the final cell in each notebook to launch the app.
Acknowledgment
This work follows the PyCaret official documentation and is independently implemented for academic submission under Dr. Elena Novak's evaluation framework.
