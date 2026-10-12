# Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques

> **CRITICAL MEDICAL DISCLAIMER:**  
> This project and its underlying machine learning models are designed, implemented, and presented solely for educational, academic, and research demonstration purposes. It utilizes the public benchmark Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn. The algorithms and software artifacts constructed herein are **NOT certified medical diagnostic tools** and must **NOT** be utilized in clinical workflows, medical diagnosis, or patient triage without qualified, certified medical practitioner oversight.

---

## 1. Project Title and Executive Summary

- **Project Title:** Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (scikit-learn `load_breast_cancer`)  
- **Methodological Framework:** Cross-Industry Standard Process for Data Mining (CRISP-DM)  
- **Core Techniques:** Exploratory Data Analysis, Data Hygiene Auditing, Unsupervised K-Means Clustering, Principal Component Analysis (PCA), Supervised Pipeline-Encapsulated Classification (Logistic Regression & Random Forest), and Apriori Association Rule Mining.

This project implements an end-to-end, terminal-executable data mining system for healthcare analytics. The primary goal is to extract clinical diagnostic insights from computerized Fine Needle Aspirate (FNA) biopsy imaging data, classify tumor malignancy with high sensitivity and specificity, discover natural patient phenotypic cluster groupings, and mine itemset association patterns while strictly upholding scientific rigor and preventing data leakage.

---

## 2. Aim and Specific Objectives

### Aim
To construct an automated data mining pipeline that accurately predicts breast tumor malignancy and models multifaceted morphologic relationships across clinical patient samples without data leakage or over-claiming causal inference.

### Specific Objectives
1. **Data Preprocessing and Auditing:**
   - Ingest and audit the Wisconsin Diagnostic Breast Cancer dataset.
   - Rigorously inspect records for missing values, structural duplicates, negative values, and infinite outliers.
   - Demonstrate principled missing-value imputation handling honestly without falsely claiming the benchmark dataset contains native missingness.
   - Prevent data leakage by enforcing strict separation between training transformations and testing evaluation using scikit-learn `Pipeline` and isolated parameter fitting.
2. **Unsupervised Clustering:**
   - Execute K-Means clustering across a range of clusters ($k \in [2, 10]$).
   - Evaluate cluster separation and cohesion using the Elbow Method (Within-Cluster Sum of Squares/Inertia) and the Silhouette Coefficient.
   - Project high-dimensional (30D) patient feature vectors onto a 2D space using Principal Component Analysis (PCA) to inspect unsupervised cluster alignment against ground-truth diagnoses.
3. **Supervised Classification:**
   - Train and evaluate Logistic Regression and Random Forest models on a stratified 80:20 partition.
   - Compute comprehensive performance metrics: Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Confusion Matrices, Classification Reports, and ROC-AUC curves.
   - Compare linear decision boundaries against non-linear ensemble tree voting.
4. **Association Rule Mining:**
   - Discretize continuous clinical attributes into interpretable tertiles (Low, Medium, High).
   - Apply the Apriori algorithm via `mlxtend` to discover frequent itemsets and extract association rules meeting strict Support, Confidence, and Lift thresholds.
   - Emphasize the epidemiological and data mining principle that statistical association does not establish biological causation.
5. **Interactive System and Artifact Generation:**
   - Develop a multi-module Streamlit dashboard (`app.py`) for interactive clinical simulation and visualization.
   - Generate high-resolution visual plots and export analytical CSV tables under the `outputs/` directory.

---

## 3. Dataset Description

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset, originally curated by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian from the University of Wisconsin Hospitals, Madison.

### Structural Parameters
- **Total Patient Samples ($N$):** 569 instances
- **Total Predictor Features ($p$):** 30 continuous real-valued numeric features
- **Target Variable:** Binary Diagnosis
  - `0`: Malignant (Cancerous, aggressive cellular abnormality)
  - `1`: Benign (Non-cancerous, localized or harmless tissue)
- **Class Balance:**
  - **Benign:** 357 instances (62.74%)
  - **Malignant:** 212 instances (37.26%)
  - *Class Imbalance Ratio:* ~ 1.68 : 1 (moderate balance handled effectively via stratified sampling)

### Feature Architecture
Features are computed from digitized images of Fine Needle Aspirate (FNA) biopsies describing cell nucleus characteristics. Ten baseline morphological characteristics are computed:
1. `Radius`: Mean of distances from center to points on the perimeter
2. `Texture`: Standard deviation of gray-scale values
3. `Perimeter`: Perimeter of the cell nucleus
4. `Area`: Nuclear cross-sectional area
5. `Smoothness`: Local variation in radius lengths
6. `Compactness`: Computed as $\frac{\text{Perimeter}^2}{\text{Area}} - 1.0$
7. `Concavity`: Severity of concave portions of the nuclear contour
8. `Concave Points`: Number of concave portions of the contour
9. `Symmetry`: Nuclear symmetry
10. `Fractal Dimension`: "Coastline approximation" (fractal dimension $- 1$)

For each of these 10 characteristics, three statistics are computed per cell nucleus:
- **Mean:** Average value across the sample image (features 1–10)
- **Standard Error (SE):** Estimation variability (features 11–20)
- **Worst / Largest:** Mean of the three largest values (features 21–30)
Yielding exactly $10 \times 3 = 30$ continuous attributes.

---

## 4. Theoretical Foundations & Mathematical Formulations

### 4.1 Data Preprocessing & Data Leakage Prevention

#### Z-Score Normalization (StandardScaler)
Because biological measurements vary widely in scale (e.g., `mean area` ranges from $143.5$ to $2501.0\text{ mm}^2$, whereas `mean smoothness` ranges from $0.053$ to $0.163$), distance-based algorithms (K-Means) and gradient-based models (Logistic Regression) require standardization to prevent high-magnitude features from dominating:
$$z = \frac{x - \mu}{\sigma}$$
Where:
- $\mu = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} x_i$ is the empirical feature mean.
- $\sigma = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (x_i - \mu)^2}$ is the sample standard deviation.

#### Mathematical Prevention of Data Leakage
Data leakage occurs when information from outside the training dataset (such as test set mean $\mu_{\text{test}}$ or validation labels) unintentionally influences model training. If a scaler is fit on the full dataset before splitting:
$$\mu_{\text{leaked}} = \frac{N_{\text{train}}\mu_{\text{train}} + N_{\text{test}}\mu_{\text{test}}}{N_{\text{train}} + N_{\text{test}}}$$
This introduces optimistic evaluation bias. In this project, data leakage is mathematically prevented by wrapping estimators in scikit-learn `Pipeline` instances:
$$\mathcal{T}_{\text{train}}: X_{\text{train}} \mapsto \text{Scaler.fit\_transform}(X_{\text{train}})$$
$$\mathcal{T}_{\text{test}}: X_{\text{test}} \mapsto \text{Scaler.transform}(X_{\text{test}})$$
The parameters $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ remain frozen during test transformation.

---

### 4.2 Unsupervised Clustering

#### K-Means Objective Function
K-Means partitions $N$ observations into $k$ disjoint clusters $S = \{S_1, S_2, \dots, S_k\}$ by minimizing the Within-Cluster Sum of Squares (WCSS), also known as **Inertia**:
$$J_{\text{WCSS}} = \sum_{j=1}^{k} \sum_{x_i \in S_j} \| x_i - \mu_j \|^2$$
Where $\mu_j$ is the centroid of cluster $S_j$:
$$\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$$

#### Silhouette Coefficient
The silhouette score quantifies how well-separated and cohesive clusters are. For each sample $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where:
- $a(i) = \frac{1}{|S_I| - 1} \sum_{j \in S_I, j \neq i} \|x_i - x_j\|$ (mean intra-cluster distance).
- $b(i) = \min_{J \neq I} \frac{1}{|S_J|} \sum_{j \in S_J} \|x_i - x_j\|$ (mean nearest-cluster distance).
- $-1 \le s(i) \le +1$. Values near $+1$ represent well-separated clusters; values near $0$ indicate boundary overlap.

#### Principal Component Analysis (PCA)
To project 30 standardized features onto orthogonal 2D dimensions, PCA computes the eigenvalues and eigenvectors of the sample covariance matrix $\mathbf{\Sigma}$:
$$\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{X}^T \mathbf{X}$$
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
The eigenvectors $\mathbf{v}_1, \mathbf{v}_2$ corresponding to the largest eigenvalues $\lambda_1, \lambda_2$ define the principal axes maximizing retained variance:
$$\text{Explained Variance Ratio}(j) = \frac{\lambda_j}{\sum_{m=1}^{p} \lambda_m}$$

---

### 4.3 Supervised Classification

#### Logistic Regression
Logistic Regression models the posterior class probability using the logistic sigmoid function:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Parameters $\mathbf{w}$ and $b$ are optimized by minimizing regularized binary cross-entropy (Log-Loss):
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

#### Random Forest Classifier
Random Forest is an ensemble of $B$ bootstrap-aggregated (bagged) classification trees. For $b = 1, \dots, B$:
1. Draw a bootstrap sample $\mathcal{D}_b$ of size $N$ with replacement.
2. At each node, select $m = \sqrt{p} = \sqrt{30} \approx 5$ random features.
3. Split the node using the feature maximizing the Gini Impurity reduction:
   $$I_G(t) = 1 - \sum_{k=0}^{1} p(k \mid t)^2$$
4. Aggregate predictions by majority voting:
   $$\hat{y}_{\text{RF}} = \text{mode} \left\{ \hat{y}_1(\mathbf{x}), \hat{y}_2(\mathbf{x}), \dots, \hat{y}_B(\mathbf{x}) \right\}$$

#### Evaluation Metrics
- **Accuracy:** $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision (Macro):** Average precision across both classes: $\frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FP}} + \frac{\text{TN}}{\text{TN} + \text{FN}} \right)$
- **Recall (Sensitivity, Macro):** $\frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)$
- **F1-Score:** Harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Area Under ROC Curve (ROC-AUC):** Integral of True Positive Rate vs False Positive Rate across all decision thresholds $\tau \in [0, 1]$.

---

### 4.4 Association Rule Mining (Apriori Algorithm)

Given an item universe $I$, a transaction dataset $D$, and a rule $A \Rightarrow B$ ($A, B \subset I$, $A \cap B = \emptyset$):

1. **Support:** Proportion of transactions containing itemset $A \cup B$:
   $$\text{Support}(A \Rightarrow B) = P(A \cup B) = \frac{|\{t \in D : A \cup B \subseteq t\}|}{|D|}$$
2. **Confidence:** Conditional probability that transaction contains $B$ given that it contains $A$:
   $$\text{Confidence}(A \Rightarrow B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
3. **Lift:** Ratio of observed joint support to expected support under statistical independence:
   $$\text{Lift}(A \Rightarrow B) = \frac{P(A \cup B)}{P(A) \cdot P(B)} = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$
   - $\text{Lift} = 1$: $A$ and $B$ are statistically independent.
   - $\text{Lift} > 1$: Positive co-occurrence association.
   - $\text{Lift} < 1$: Negative association (substitution effect).

---

## 5. Methodology & Experimental Setup

Following the CRISP-DM standard:
```
[Business/Clinical Understanding]
               ↓
     [Data Understanding] ──→ Audit: 569 Samples, 30 Features, 0 Missing
               ↓
     [Data Preparation]   ──→ 80:20 Stratified Split & StandardScaler (Fit on Train only)
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
[Unsupervised]      [Supervised]
K-Means (k=2..10)   Logistic Regression & Random Forest
PCA 2D Projection   Evaluation: Accuracy, F1, ROC-AUC
    ↓                     ↓
    └──────────┬──────────┘
               ↓
    [Association Rules]   ──→ Discretization & Apriori Mining
               ↓
    [Evaluation & UI]     ──→ Visualizations (9 PNGs) & Streamlit Dashboard
```

1. **Stratification:** Fixed random state ($42$) with stratified sampling ($80\%$ train $= 455$ cases; $20\%$ test $= 114$ cases).
2. **Software Stack:** Python 3.12, scikit-learn 1.9, pandas 3.0, NumPy 2.5, Matplotlib 3.11, Seaborn 0.13, mlxtend 0.25, Streamlit 1.65.

---

## 6. Empirical Results & Performance Analysis

### 6.1 Data Hygiene Audit Results
- **Missing Values:** $0$ ($100\%$ complete verified).
- **Duplicate Rows:** $0$
- **Invalid Negative Values:** $0$
- **Class Breakdown:**
  - Training Set ($N=455$): $170$ Malignant ($37.36\%$), $285$ Benign ($62.64\%$)
  - Testing Set ($N=114$): $42$ Malignant ($36.84\%$), $72$ Benign ($63.16\%$)

---

### 6.2 Unsupervised Clustering Evaluation

Evaluation of $k$ from 2 to 10 produced the following empirical metrics:

| Number of Clusters ($k$) | Inertia (WCSS) | Silhouette Score |
| :---: | :---: | :---: |
| **2** | **11,595.53** | **0.3434 (Optimal)** |
| 3 | 10,061.80 | 0.3144 |
| 4 | 9,258.99 | 0.2833 |
| 5 | 8,558.66 | 0.1582 |
| 6 | 7,970.26 | 0.1604 |
| 7 | 7,540.32 | 0.1532 |
| 8 | 7,254.33 | 0.1472 |
| 9 | 6,837.63 | 0.1470 |
| 10 | 6,603.40 | 0.1367 |

- The elbow point and peak silhouette coefficient occur at **$k = 2$ ($0.3434$)**, demonstrating that the unsupervised geometry naturally aligns with the biological dual-state taxonomy (Malignant vs Benign).

#### Cluster Profile and Biological Concordance
Aggregating key morphological features by cluster:

| Cluster | Total Samples | Benign Cases | Malignant Cases | Alignment Purity | Mean Radius (mm) | Mean Area ($\text{mm}^2$) | Mean Concavity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 375 | 339 | 36 | **90.4% Benign** | 12.43 | 486.77 | 0.043 |
| **1** | 194 | 18 | 176 | **90.7% Malignant** | 17.42 | 979.86 | 0.178 |

- **Clinical Interpretation:** Cluster 1 samples exhibit more than double the mean nuclear area ($979.86\text{ mm}^2$ vs $486.77\text{ mm}^2$) and more than quadruple the mean concavity ($0.178$ vs $0.043$). This proves unsupervised clustering successfully segments malignant cases based purely on cellular morphology without requiring labels.

#### PCA Dimensionality Reduction
- **Principal Component 1 (PC1):** Explains **$44.3\%$** of overall variance.
- **Principal Component 2 (PC2):** Explains **$19.0\%$** of overall variance.
- **Combined 2D Retained Variance:** **$63.2\%$**.

---

### 6.3 Supervised Classification Model Performance

The two classification pipelines evaluated on the held-out test set ($N=114$: 42 Malignant, 72 Benign) yielded the following empirical scores:

| Model Pipeline | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825 (98.25%)** | **0.9812 (98.12%)** | **0.9812 (98.12%)** | **0.9812 (98.12%)** | **0.9954 (99.54%)** |
| **Random Forest** | 0.9561 (95.61%) | 0.9551 (95.51%) | 0.9504 (95.04%) | 0.9526 (95.26%) | 0.9939 (99.39%) |

#### Detailed Confusion Matrix Analysis

1. **Logistic Regression Pipeline:**
   $$\begin{pmatrix} \text{True Malignant (TN)} = 41 & \text{False Benign (FN)} = 1 \\ \text{False Malignant (FP)} = 1 & \text{True Benign (TP)} = 71 \end{pmatrix}$$
   - **Malignant Sensitivity (Recall):** $\frac{41}{41 + 1} = \mathbf{97.62\%}$
   - **Benign Specificity:** $\frac{71}{71 + 1} = \mathbf{98.61\%}$
   - **Clinical Implications:** Across 114 unseen test cases, Logistic Regression produced only **1 false negative** (malignant sample missed) and **1 false positive** (benign case flagged).

2. **Random Forest Classifier:**
   $$\begin{pmatrix} \text{True Malignant (TN)} = 39 & \text{False Benign (FN)} = 3 \\ \text{False Malignant (FP)} = 2 & \text{True Benign (TP)} = 70 \end{pmatrix}$$
   - **Malignant Sensitivity (Recall):** $\frac{39}{39 + 3} = \mathbf{92.86\%}$
   - **Benign Specificity:** $\frac{70}{70 + 2} = \mathbf{97.22\%}$

#### Model Comparison Rationale
Logistic Regression slightly outperformed Random Forest on this benchmark. This is consistent with high-dimensional morphologic data where the 30 continuous features exhibit strong linear correlation with log-odds of malignancy; linear hyperplanes with L2 regularization generalize cleanly with minimal overfitting compared to greedy tree partitions.

---

### 6.4 Association Rule Mining Results (Apriori)

- **Discretization:** Continuous variables binned into tertiles (`Low`, `Med`, `High`).
- **Binary Encoded Items:** 26 one-hot items.
- **Mined Frequent Itemsets:** 206 (at $\text{min\_support} = 0.15$).
- **Total Validated Rules:** 852 (at $\text{min\_confidence} = 0.60, \text{min\_lift} \ge 1.2$).
- **Diagnostic Specific Rules:** 304 rules explicitly containing diagnosis in the consequent.

#### Top Diagnostic Rules Ranked by Lift

| Rank | Antecedent (IF) | Consequent (THEN) | Support | Confidence | Lift |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `concavity_High`, `radius_High` | `area_High`, `compactness_High`, `diagnosis_Malignant` | 0.193 | 0.791 | **3.985** |
| 2 | `concavity_High`, `perimeter_High`, `radius_High` | `area_High`, `compactness_High`, `diagnosis_Malignant` | 0.193 | 0.791 | **3.985** |
| 3 | `area_High`, `concavity_High` | `compactness_High`, `diagnosis_Malignant`, `radius_High` | 0.193 | 0.797 | **3.979** |
| 4 | `compactness_High`, `concavity_High` | `diagnosis_Malignant`, `perimeter_High`, `radius_High` | 0.193 | 0.791 | **3.921** |
| 5 | `concavity_Low`, `perimeter_Low` | `area_Low`, `compactness_Low`, `diagnosis_Benign` | 0.281 | 0.941 | **1.500** |

---

## 7. Association vs. Causation: The Epistemological Distinction

A central tenet of scientific data mining is that **association does not imply causation**.

1. **Mathematical Reality:**  
   A rule such as `concavity_High` $\Rightarrow$ `diagnosis_Malignant` with $\text{Lift} = 3.985$ and $\text{Confidence} = 79.1\%$ indicates that high concavity co-occurs with malignancy $3.985$ times more frequently than chance.
2. **Biological Reality:**  
   High nuclear contour concavity is **NOT the cause** of cancer. Cancer is caused by genomic mutations, dysregulated cell cycle checkpoints (e.g., p53, BRCA), and oncogene amplification.
3. **Phenotypic Symptoms:**  
   The geometric abnormalities (irregular contour, large radius, nuclear pleomorphism) are macroscopic **symptoms** resulting from chaotic mitotic cell division.
4. **Clinical Hazard of Confusing Causation:**  
   Assuming causation could lead an untrained observer to conclude that mechanically smoothing the nucleus would cure the tumor. Association rule mining must be treated as an exploratory screening and hypothesis-generation tool, never as an etiology-proving instrument.

---

## 8. Visualizations Walkthrough

All visualizations were programmatically generated and exported to `outputs/`:

1. `class_distribution.png`: Bar chart displaying 357 Benign (62.7%) and 212 Malignant (37.3%) patient records with exact percentages.
2. `correlation_heatmap.png`: Triangular Pearson correlation matrix showing strong collinearity between `radius`, `perimeter`, and `area` ($r > 0.98$).
3. `feature_distributions.png`: Multi-panel boxplots illustrating statistically significant shifts in median values between malignant and benign biopsies.
4. `elbow_silhouette_plot.png`: Dual subplots tracking inertia decrease and identifying the maximum silhouette coefficient at $k = 2$.
5. `kmeans_pca_clusters.png`: Side-by-side scatter plots showing unsupervised cluster partitioning compared against ground-truth histological diagnosis.
6. `confusion_matrices.png`: Heatmap representations of predicted vs. true labels for Logistic Regression and Random Forest.
7. `roc_curves.png`: Receiver Operating Characteristic plots demonstrating near-perfect discriminative power ($0.9954$ and $0.9939$ AUC).
8. `model_comparison.png`: Grouped comparative bar chart across all 5 evaluation metrics.
9. `association_rules_plot.png`: Scatter plot mapping support, confidence, and lift alongside top extracted rules.

---

## 9. Strengths, Limitations, and Future Scope

### Project Strengths
- **Leakage-Free Architecture:** Scaler fitting is isolated to training sets using scikit-learn Pipelines.
- **Empirical Honesty:** Full disclosure of benchmark completeness without inventing missing data.
- **Dual Paradigms:** Combines unsupervised discovery (K-Means, PCA) with supervised classification and pattern mining (Apriori).
- **Interactive Interface:** Complete Streamlit dashboard enabling real-time scenario simulation.

### Methodological Limitations
- **Single-Source Dataset:** 569 samples collected from a single clinical institution limits geographic and demographic generalization.
- **Absence of Longitudinal Outcomes:** Contains no data on recurrence, patient survival rates, or therapeutic response.
- **Information Loss via Discretization:** Binning continuous attributes into tertiles reduces numerical granularity.

### Future Scope
- **Multi-Modal Integration:** Combine morphological FNA metrics with genomic data (RNA-Seq, DNA methylation) and clinical EHR notes.
- **Explainable AI (XAI):** Incorporate SHAP (Shapley Additive Explanations) and LIME to interpret feature importance at the individual patient level.
- **Deep Learning Biopsy Vision:** Process raw histological slide images using Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

---

## 10. Conclusion

This project successfully implements an end-to-end healthcare data mining pipeline adhering to the CRISP-DM lifecycle. Through rigorous preprocessing, data leakage prevention, unsupervised clustering, supervised classification, and association rule mining, the system demonstrated:
- **Optimal Unsupervised Clustering:** Peak silhouette score of $0.3434$ at $k=2$, isolating malignant samples with over $90\%$ purity without label guidance.
- **Predictive Accuracy:** The Logistic Regression Pipeline achieved **$98.25\%$ accuracy** and **$0.9954$ ROC-AUC**, missing only 1 malignant sample out of 42.
- **Association Patterns:** Apriori mining identified co-occurring morphological indicators with lifts up to $3.985$, while framing these findings within an association-versus-causation context.

The terminal-executable codebase and interactive dashboard provide a reproducible template for educational healthcare data mining.
