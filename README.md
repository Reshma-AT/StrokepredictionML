# Stroke Prediction Model (Binary Classification)

This repository contains a machine learning pipeline designed to predict the likelihood of a patient suffering a stroke based on demographic, lifestyle, and medical history features. 

Owing to the heavily imbalanced nature of medical diagnostic data, this project focuses closely on optimizing model behavior beyond simple accuracy, utilizing advanced evaluation metrics and gradient boosting methods.

---

## 📌 Project Overview & Challenge

* **The Problem:** The dataset exhibits a significant class imbalance:
  * **No Stroke (Target = 0):** 4,861 patients (~95.13%)
  * **Stroke (Target = 1):** 249 patients (< 4.87%)
* **The Metric Pitfall:** A trivial predictor that defaults to predicting `0` for every patient would mathematically achieve **95.13% accuracy** while offering zero clinical or discriminative value.
* **The Strategy:** While tracking baseline accuracy, this project prioritizes **Area Under the ROC Curve (AUC)** and **Precision-Recall AUC (AUCPR)** to establish robust predictive power.

---

## 🛠️ Tech Stack & Libraries

* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`, `statsmodels` (for Mosaic mapping)
* **Machine Learning Engine:** `H2O.ai` (H2O Gradient Boosting Estimator)

---

## 📖 Pipeline & Workflow

### 1. Data Cleansing & Feature Engineering
* **Imputation Insight:** Missing values detected in the `bmi` feature were strategically imputed with a placeholder value (`-99`). Univariate exploration revealed a critical pattern: **~20% of patients with missing BMI data had suffered a stroke**, rendering the "missingness" highly predictive rather than random.
* **Binning Strategy:** Quantile-based and custom binning were introduced on numeric features (`age`, `avg_glucose_level`, `bmi`) to build explanatory Mosaic visualizations.

### 2. Exploratory Data Analysis (EDA) Highlights
* **Numerical Correlates:** Risk escalates notably with advancing `age` and elevated `avg_glucose_level` (indicative of diabetic risk).
* **Categorical Correlates:** Presence of hypertension and heart disease drastically increases stroke occurrence. Interestingly, univariate analysis showed a minimal variance between current smokers and individuals who have never smoked.

### 3. Model Training
* Built using an **H2O Gradient Boosting Machine (GBM)** utilizing a 70/30 stratified Train/Test split.
* Configured with **5-Fold Cross-Validation** and early stopping configurations based on validation AUC metrics to limit overfitting.

---

## 📊 Model Evaluation & Calibration

### Threshold Optimization
Relying on the default maximum $F_1$-score threshold yielded a major discrepancy between actual and predicted positives. To correct for clinical utility, the classification threshold was manually tuned to a symmetric boundary of **0.148**:

| Evaluation Set | Threshold Value | Achieved Accuracy | Actual Positives | Predicted Positives |
| :--- | :--- | :--- | :--- | :--- |
| **Training Data** | 0.1478 | 93.95% | 184 | 185 |
| **Cross-Validation** | 0.1477 | 92.11% | 184 | 181 |
| **Test Set** | 0.1480 | **93.16%** | **65** | **69** |

### High & Low Risk Calibration
Sorting predictions on the unseen test set demonstrates excellent model alignment with real-world outcomes:
* **Highest 20 Patients Ranked by Risk:** The model predicted an aggregate frequency of `6.69` expected strokes; the actual target contained `5` real-world stroke events. 
* **Lowest 20 Patients Ranked by Risk:** The model predicted a negligible baseline frequency of `0.42` expected cases; the group contained `0` actual stroke events (predominantly composed of children and young adults with no compounding medical histories).

### Feature Impact (SHAP Interpretability)
Feature significance plots generated via TreeSHAP confirm that **Age** acts as the primary driver of stroke susceptibility, followed sequentially by blood glucose concentration and underlying cardiac health issues.

---

## 🚀 How to Run Local Cluster

Ensure you have Java 11 or higher installed on your environment to run the backend H2O cluster instance.

```python
import h2o
# Initialize local cluster with 4 threads
h2o.init(max_mem_size='12G', nthreads=4)


## ⚙️ Installation & Setup

### Prerequisites
Before running the code, ensure you have the following installed on your machine:
* **Python 3.8+**
* **Java Runtime Environment (JRE) or JDK (Version 11 or higher)** — *Mandatory because the Python `h2o` library runs on a local in-memory Java virtual machine cluster backend.*
* **pip** (Python package installer)

### 1. Clone Repository
Open your terminal and run the following commands to download the project locally:
```bash
git clone [https://github.com/your-username/stroke-prediction-model.git](https://github.com/your-username/stroke-prediction-model.git)
cd stroke-prediction-model

# Create the environment
python -m venv venv

# Windows Activation
venv\Scripts\activate

# Mac/Linux Activation
source venv/bin/activate

pip install pandas numpy matplotlib seaborn statsmodels h2o scikit-learn streamlit

##Train the Model

python train_model.py

##Launch the interactive web platform to run diagnostic assessments interactively.

streamlit run app.py
