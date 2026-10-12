# Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **⚠️ MEDICAL & EDUCATIONAL DISCLAIMER:**  
> This project is developed exclusively for educational and academic data mining research using the Breast Cancer Wisconsin (Diagnostic) benchmark dataset. **It is NOT a medical diagnostic tool** and must not be used for real-world patient triage or clinical treatment decisions.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Technology Stack](#technology-stack)
4. [Installation and Environment Setup](#installation-and-environment-setup)
5. [How to Run (VS Code Terminal)](#how-to-run-vs-code-terminal)
6. [Pipeline Modules](#pipeline-modules)
7. [Empirical Results](#empirical-results)
8. [Association vs Causation](#association-vs-causation)
9. [Streamlit Interactive Dashboard](#streamlit-interactive-dashboard)

---

## 1. Project Overview

This project implements an end-to-end, terminal-executable Data Mining system adhering to the **CRISP-DM** lifecycle. The dataset utilized is the public **Breast Cancer Wisconsin (Diagnostic)** dataset from scikit-learn ($N=569$, $p=30$ continuous features).

The pipeline covers:
- **Data Preprocessing & Auditing:** Automated hygiene checks (nulls, duplicates, outliers), honest missing-value demonstration, and leakage-free scaling.
- **Unsupervised Clustering:** K-Means clustering, Elbow Method (Inertia), Silhouette Score analysis, and 2D PCA projection.
- **Supervised Classification:** Leakage-free scikit-learn `Pipeline` models (Logistic Regression & Random Forest) with 80:20 stratified splitting and metric evaluation.
- **Association Rule Mining:** Continuous feature discretization into tertiles and Apriori frequent itemset mining with support, confidence, and lift metrics.
- **Interactive Web App:** A multi-page Streamlit dashboard featuring live patient simulation.

---

## 2. Directory Structure

```
healthcare_data_mining/
│
├── app.py                     # Streamlit interactive dashboard application
├── data_processing.py         # Data loading, hygiene audit, and leakage-free scaling
├── clustering.py              # K-Means clustering, Elbow method, Silhouette, PCA
├── classification.py          # Supervised classification pipelines & model comparison
├── association_rules.py       # Apriori frequent itemset and association rule mining
├── visualization.py           # Master visualization generator (all 9 plots)
├── requirements.txt           # Python dependency requirements
├── README.md                  # Detailed project documentation and run instructions
├── report.md                  # Comprehensive academic and technical research report
│
└── outputs/                   # Generated evaluation tables and high-res visualizations
    ├── association_rules.csv
    ├── association_rules_plot.png
    ├── class_distribution.png
    ├── cleaned_data.csv
    ├── clustering_metrics.csv
    ├── cluster_profile.csv
    ├── confusion_matrices.png
    ├── correlation_heatmap.png
    ├── data_summary.txt
    ├── elbow_silhouette_plot.png
    ├── feature_distributions.png
    ├── kmeans_pca_clusters.png
    ├── model_comparison.csv
    ├── model_comparison.png
    └── roc_curves.png
```

---

## 3. Technology Stack

- **Language:** Python 3.12
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`
- **Association Rule Mining:** `mlxtend`
- **Interactive UI:** `streamlit`

---

## 4. Installation and Environment Setup

Open the terminal in **Visual Studio Code** (`Ctrl + \`` or `Terminal` -> `New Terminal`).

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\prana\OneDrive\Desktop\Projects\Data Mining"
```

### Step 2: Install Required Libraries
```powershell
python -m pip install -r requirements.txt
```

---

## 5. How to Run (VS Code Terminal)

Execute each module directly through the VS Code terminal:

### 1. Data Preprocessing & Hygiene Audit
```powershell
python data_processing.py
```
*Outputs: `outputs/cleaned_data.csv` and `outputs/data_summary.txt`.*

### 2. Unsupervised Clustering & PCA
```powershell
python clustering.py
```
*Outputs: `outputs/elbow_silhouette_plot.png`, `outputs/kmeans_pca_clusters.png`, `outputs/clustering_metrics.csv`, and `outputs/cluster_profile.csv`.*

### 3. Supervised Classification Models
```powershell
python classification.py
```
*Outputs: `outputs/confusion_matrices.png`, `outputs/roc_curves.png`, `outputs/model_comparison.png`, and `outputs/model_comparison.csv`.*

### 4. Association Rule Mining (Apriori)
```powershell
python association_rules.py
```
*Outputs: `outputs/association_rules.csv` and `outputs/association_rules_plot.png`.*

### 5. Generate All Visualizations At Once
```powershell
python visualization.py
```
*Regenerates and verifies all 9 high-resolution images in `outputs/`.*

### 6. Launch the Interactive Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
*Opens the interactive clinical dashboard in your web browser (default: `http://localhost:8501`).*

---

## 6. Pipeline Modules

### `data_processing.py`
- Ingests the Wisconsin Breast Cancer dataset ($N=569$).
- Checks missing values ($0$), duplicates ($0$), and invalid values ($0$).
- Demonstrates `SimpleImputer(strategy='mean')` handling on simulated missing data.
- Enforces data leakage prevention: fits `StandardScaler` strictly on training data (`X_train`) within an 80:20 stratified split.

### `clustering.py`
- Evaluates K-Means clustering across $k \in [2, 10]$.
- Generates the Elbow curve (Inertia) and Silhouette score curve.
- Projects samples onto 2D PCA components ($\text{PC1}=44.3\%$, $\text{PC2}=19.0\%$ variance).
- Discovers unsupervised cluster purity of $>90\%$ matching true biological labels.

### `classification.py`
- Wraps estimators in scikit-learn `Pipeline` objects.
- Evaluates **Logistic Regression** and **Random Forest**.
- Computes Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC, and Confusion Matrices.

### `association_rules.py`
- Discretizes continuous numerical features into tertiles (`Low`, `Med`, `High`).
- Uses `mlxtend.frequent_patterns.apriori` and `association_rules`.
- Computes Support, Confidence, Lift, Leverage, and Conviction.
- Filters and ranks diagnostic-specific rules.

### `visualization.py`
- Master script that plots and saves high-resolution (300 DPI) figures for all EDA, clustering, classification, and association results.

### `app.py`
- Streamlit interactive application featuring:
  1. Executive Dataset Overview & Metrics
  2. Data Hygiene & Leakage Prevention Audit
  3. K-Means Clustering & PCA 2D Space
  4. Classification Performance & ROC Curves
  5. Interactive Association Rule Filters
  6. **Interactive Patient Risk Predictor (Live Simulation)**
  7. Comprehensive Academic Report View

---

## 7. Empirical Results

Results obtained from the held-out test set ($N=114$, Stratified 80:20):

| Model Pipeline | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.25%** | **98.12%** | **98.12%** | **98.12%** | **0.9954** |
| **Random Forest** | **95.61%** | **95.51%** | **95.04%** | **95.26%** | **0.9939** |

### Unsupervised Clustering Results:
- **Optimal Silhouette Score:** `0.3434` at `k = 2`.
- **Cluster 0:** 375 samples ($90.4\%$ Benign).
- **Cluster 1:** 194 samples ($90.7\%$ Malignant).

---

## 8. Association vs Causation

```
⚠️ IMPORTANT SCIENTIFIC PRINCIPLE:
   Statistical Association  ≠  Biological Causation
```
In association rule mining:
$$\{ \text{concavity\_High}, \text{radius\_High} \} \Longrightarrow \{ \text{diagnosis\_Malignant} \} \quad (\text{Lift} = 3.985)$$
This indicates strong statistical co-occurrence within biopsy imaging samples. It does **not** prove high concavity causes cancer; both are morphological consequences of genetic and biochemical cellular deregulation.

---

## 9. Streamlit Interactive Dashboard

To start the interactive dashboard:
```powershell
python -m streamlit run app.py
```
Features:
- **Live scenario simulation:** Test "Typical Benign", "Typical Malignant", or adjust patient sliders to observe real-time risk predictions and probability gauges.
- **Rule filtering:** Filter association rules dynamically by Minimum Lift, Confidence, and Target Diagnosis.
- **Visuals gallery:** View all 9 high-resolution diagnostic charts directly inside the web UI.
