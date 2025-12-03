# Healthcare Provider Fraud Detection Project

##  Project Overview
Healthcare fraud is a significant issue leading to billions of dollars in losses annually. This project applies machine learning techniques to detect potential fraud among healthcare providers. By analyzing claims data, beneficiary details, and provider patterns, we aim to build a predictive model that identifies "Potentially Fraudulent" providers for further investigation.

## Repository Structure

The project is structured into three sequential notebooks and supporting data files:

### Notebooks
* **`01_data_exploration_and_feature_engineering.ipynb`**
    * **Purpose:** Data cleaning, merging, and exploratory data analysis (EDA).
    * **Key Actions:** Aggregates claim-level data (Inpatient/Outpatient) to the Provider level. Performs correlation analysis to identify key drivers of fraud (e.g., volume vs. reimbursement). Handles missing values and class imbalance.
* **`02_modeling.ipynb`**
    * **Purpose:** Model training and selection.
    * **Key Actions:** Trains three classifiers: Logistic Regression, Random Forest, and Gradient Boosting. Compares performance using cross-validation. Selects the **Gradient Boosting Classifier** as the best performer.
* **`03_evaluation.ipynb`**
    * **Purpose:** In-depth testing and error analysis.
    * **Key Actions:** Evaluates the best model on the hold-out test set (`X_test`, `y_test`). Analyzes False Positives vs. False Negatives and provides business recommendations based on financial cost.

### Data Files
* `provider_features_for_modeling.csv`: The final processed dataset used for training.
* `X_test_scaled.csv` & `y_test.csv`: The hold-out dataset used for final evaluation.
* `feature_names.csv`: A list of the input features used by the model.
* `best_model.pkl`: The serialized, trained Gradient Boosting model (ready for deployment).

##  Installation & Usage

### Prerequisites
* Python 3.8 or higher
* Jupyter Notebook or JupyterLab

### Requirements
Install the necessary Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
