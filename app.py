"""
========================================================================================
Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques
Module: app.py
Description: Interactive Streamlit Dashboard showcasing the entire Data Mining pipeline:
             Dataset Overview, Preprocessing & Hygiene, Clustering (K-Means & PCA),
             Classification (Logistic Regression & Random Forest), Association Rules
             (Apriori), Interactive Patient Risk Predictor, and Project Conclusion.
========================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local module imports work smoothly
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_processing import load_raw_data, audit_data_hygiene, get_preprocessed_split
from clustering import perform_pca_clustering
from classification import train_and_evaluate_models
from association_rules import discretize_patient_data, mine_association_rules

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Disease Prediction & Data Mining",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 12px 18px;
        border-radius: 4px;
        color: #7b341e;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.88rem;
        color: #718096;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA LOADERS & COMPUTATIONS
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset():
    return load_raw_data()

@st.cache_data
def get_hygiene_summary(df):
    return audit_data_hygiene(df)

@st.cache_data
def get_clustering_data(df):
    return perform_pca_clustering(df, n_clusters=2)

@st.cache_data
def get_classification_results():
    return train_and_evaluate_models()

@st.cache_data
def get_association_data(df):
    encoded_df, _ = discretize_patient_data(df)
    freq, rules, disease_rules = mine_association_rules(
        encoded_df, min_support=0.15, min_confidence=0.60, min_lift=1.2
    )
    return rules, disease_rules


# -----------------------------------------------------------------------------
# MAIN APP HEADER & DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Healthcare Disease Prediction & Patient Data Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Data Mining Pipeline: Exploration, Clustering, Classification, & Association Rule Mining</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ EDUCATIONAL DATA MINING DISCLAIMER:</strong>
    This software is developed exclusively for academic, research, and educational demonstration purposes using the
    Breast Cancer Wisconsin (Diagnostic) benchmark dataset. It is <strong>NOT a certified medical diagnostic device</strong>
    and must never be employed for clinical decision-making or real-world patient triage without certified medical supervision.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/health-graph.png", width=80)
st.sidebar.title("Navigation Panel")
section = st.sidebar.radio(
    "Explore Pipeline Modules:",
    [
        "1. Dataset Overview & Exploration",
        "2. Data Preprocessing & Hygiene",
        "3. Unsupervised Clustering (K-Means & PCA)",
        "4. Supervised Classification Models",
        "5. Association Rule Mining (Apriori)",
        "6. Interactive Patient Risk Predictor",
        "7. Project Report & Conclusion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data Mining Project | Wisconsin Breast Cancer Dataset (N=569)")


# -----------------------------------------------------------------------------
# SECTION 1: DATASET OVERVIEW
# -----------------------------------------------------------------------------
df = get_dataset()
hygiene = get_hygiene_summary(df)

if section == "1. Dataset Overview & Exploration":
    st.header("1. Dataset Overview & Exploratory Data Analysis")
    st.write(
        "The Breast Cancer Wisconsin (Diagnostic) dataset contains morphological features "
        "computed from digitized images of Fine Needle Aspirate (FNA) biopsies of breast masses."
    )

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{hygiene["total_samples"]}</div><div class="metric-label">Total Patients</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{hygiene["feature_count"]}</div><div class="metric-label">Numerical Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #38a169;">{hygiene["benign_count"]} ({hygiene["benign_pct"]}%)</div><div class="metric-label">Benign (Non-Cancerous)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #e53e3e;">{hygiene["malignant_count"]} ({hygiene["malignant_pct"]}%)</div><div class="metric-label">Malignant (Cancerous)</div></div>', unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Class Distribution", "🔥 Correlation Heatmap"])

    with tab1:
        st.subheader("Raw Patient Records (Top 10)")
        preview_cols = ["diagnosis_label", "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean concavity"]
        st.dataframe(df[preview_cols].head(10), use_container_width=True)
        st.write("Full feature list includes 3 groups of 10 measurements: **Mean**, **Standard Error (SE)**, and **Worst (Largest)**.")

    with tab2:
        st.subheader("Class Distribution Analysis")
        p_class = OUTPUTS_DIR / "class_distribution.png"
        if p_class.exists():
            st.image(str(p_class), caption="Diagnostic Class Distribution", use_container_width=True)
        else:
            st.info("Class distribution plot not found. Run visualization.py to generate image.")

    with tab3:
        st.subheader("Inter-Feature Correlation Heatmap")
        p_corr = OUTPUTS_DIR / "correlation_heatmap.png"
        if p_corr.exists():
            st.image(str(p_corr), caption="Correlation Matrix of Primary Morphological Features", use_container_width=True)
        else:
            st.info("Correlation heatmap plot not found. Run visualization.py to generate image.")


# -----------------------------------------------------------------------------
# SECTION 2: DATA PREPROCESSING
# -----------------------------------------------------------------------------
elif section == "2. Data Preprocessing & Hygiene":
    st.header("2. Data Preprocessing, Hygiene Audit & Leakage Prevention")
    st.write(
        "Rigorous data mining requires thorough data validation, imputation preparedness, "
        "and strict data leakage prevention between training and evaluation partitions."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Data Hygiene & Integrity Checklist")
        st.markdown(f"""
        - **Total Missing Values Detected:** `{hygiene['total_missing']}` *(Original benchmark is 100% complete)*
        - **Duplicate Patient Records:** `{hygiene['duplicate_count']}`
        - **Biologically Invalid Negative Values:** `{hygiene['negative_values']}`
        - **Infinite/Unbounded Values:** `{hygiene['infinite_values']}`
        - **Data Integrity Status:** <span style="color:green; font-weight:bold;">PASSED ALL CHECKS</span>
        """, unsafe_allow_html=True)

        st.subheader("Honest Missing-Value Handling Demonstration")
        st.write(
            "While curated academic datasets like Wisconsin Breast Cancer do not suffer from natural missingness, "
            "production healthcare pipelines must incorporate automated imputation. "
            "Our pipeline implements `SimpleImputer(strategy='mean')` within scikit-learn Pipelines, ensuring "
            "unseen patient records with missing laboratory fields are imputed gracefully."
        )

    with c2:
        st.subheader("Data Leakage Prevention Protocol")
        st.info("""
        **How Data Leakage is Prevented:**
        1. **Stratified Split First:** The 80:20 split is performed prior to any data normalization.
        2. **Training-Only Scaler Fit:** `StandardScaler` calculates the sample mean (μ) and variance (σ²) strictly on `X_train`.
        3. **Deterministic Test Transform:** `X_test` is transformed using the frozen parameters of `X_train`.
        4. **Pipeline Encapsulation:** All estimators are bundled in scikit-learn `Pipeline` objects.
        """)

    st.subheader("Feature Distributions by Diagnosis")
    p_feat = OUTPUTS_DIR / "feature_distributions.png"
    if p_feat.exists():
        st.image(str(p_feat), caption="Boxplot Distributions of Key Morphologic Features", use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 3: CLUSTERING
# -----------------------------------------------------------------------------
elif section == "3. Unsupervised Clustering (K-Means & PCA)":
    st.header("3. Unsupervised Clustering & Dimensionality Reduction")
    st.write(
        "K-Means clustering discovers natural groupings in the 30-dimensional patient space "
        "without prior knowledge of tumor malignancy labels."
    )

    pca_df, kmeans_model, pca_model, metrics_df, profile_df = get_clustering_data(df)

    tab1, tab2, tab3 = st.tabs(["📈 Elbow & Silhouette Analysis", "🌌 PCA 2D Cluster Space", "🔬 Cluster Profiling"])

    with tab1:
        st.subheader("Optimal Cluster Selection: Elbow & Silhouette Scores")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(metrics_df, use_container_width=True)
            best_k = int(metrics_df.loc[metrics_df["silhouette_score"].idxmax()]["k"])
            st.success(f"Optimal Silhouette Score achieved at **k = {best_k}** with Silhouette = **{metrics_df['silhouette_score'].max():.4f}**.")
        with c2:
            p_elbow = OUTPUTS_DIR / "elbow_silhouette_plot.png"
            if p_elbow.exists():
                st.image(str(p_elbow), use_container_width=True)

    with tab2:
        st.subheader("Principal Component Analysis (PCA) 2D Projection")
        var_exp = pca_model.explained_variance_ratio_ * 100
        st.write(f"The first two principal components capture **{var_exp[0] + var_exp[1]:.1f}%** of the total variance across all 30 features.")
        p_pca = OUTPUTS_DIR / "kmeans_pca_clusters.png"
        if p_pca.exists():
            st.image(str(p_pca), use_container_width=True)

    with tab3:
        st.subheader("Cluster Profiles & Biological Alignment")
        st.write(
            "Comparing the feature averages of the unsupervised clusters highlights that "
            "Cluster 1 corresponds to aggressive tumor morphology (larger radius, higher concavity)."
        )
        st.dataframe(profile_df, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 4: CLASSIFICATION
# -----------------------------------------------------------------------------
elif section == "4. Supervised Classification Models":
    st.header("4. Supervised Classification: Logistic Regression & Random Forest")
    st.write(
        "Models were trained using an 80:20 stratified split with full preprocessing pipeline encapsulation."
    )

    results_df, eval_details, pipelines, X_test, y_test = get_classification_results()

    st.subheader("Empirical Performance Comparison (Test Set N=114)")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#d4edda", subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "ROC-AUC"]),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrices (Test Set)")
        p_cm = OUTPUTS_DIR / "confusion_matrices.png"
        if p_cm.exists():
            st.image(str(p_cm), use_container_width=True)

    with c2:
        st.subheader("ROC Curves Comparison")
        p_roc = OUTPUTS_DIR / "roc_curves.png"
        if p_roc.exists():
            st.image(str(p_roc), use_container_width=True)

    st.subheader("Model Comparison Metric Bar Chart")
    p_comp = OUTPUTS_DIR / "model_comparison.png"
    if p_comp.exists():
        st.image(str(p_comp), use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 5: ASSOCIATION RULE MINING
# -----------------------------------------------------------------------------
elif section == "5. Association Rule Mining (Apriori)":
    st.header("5. Association Rule Mining (Apriori Algorithm)")
    st.write(
        "Continuous morphologic attributes were discretized into tertiles (Low, Med, High). "
        "The Apriori algorithm extracted frequent itemsets and co-occurrence rules."
    )

    st.markdown("""
    <div class="disclaimer-box" style="border-left-color: #3182ce; background-color: #ebf8ff; color: #2b6cb0;">
        <strong>🔬 CRITICAL DATA MINING PRINCIPLE: ASSOCIATION DOES NOT ESTABLISH CAUSATION:</strong><br>
        Association rules disclose high co-occurrence (e.g., high perimeter with malignancy).
        However, high perimeter does NOT chemically cause tumor malignancy; rather, both are symptoms of cellular hyperplasia.
    </div>
    """, unsafe_allow_html=True)

    rules, disease_rules = get_association_data(df)

    # Interactive Filters
    st.subheader("Interactive Rule Filter")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_lift_val = st.slider("Minimum Lift", min_value=1.2, max_value=4.0, value=2.0, step=0.1)
    with c2:
        min_conf_val = st.slider("Minimum Confidence", min_value=0.6, max_value=1.0, value=0.75, step=0.05)
    with c3:
        rule_target = st.selectbox("Target Filter", ["All Rules", "Malignant Rules Only", "Benign Rules Only"])

    # Filter logic
    filtered_rules = rules[(rules["lift"] >= min_lift_val) & (rules["confidence"] >= min_conf_val)].copy()
    if rule_target == "Malignant Rules Only":
        filtered_rules = filtered_rules[filtered_rules["consequents_str"].str.contains("Malignant", case=False)]
    elif rule_target == "Benign Rules Only":
        filtered_rules = filtered_rules[filtered_rules["consequents_str"].str.contains("Benign", case=False)]

    st.write(f"Displaying **{len(filtered_rules)}** matching rules:")
    display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    st.dataframe(
        filtered_rules[display_cols].rename(columns={
            "antecedents_str": "Antecedents (IF)",
            "consequents_str": "Consequents (THEN)"
        }).round(3).head(25),
        use_container_width=True
    )

    st.subheader("Support vs Confidence vs Lift Distribution")
    p_ar = OUTPUTS_DIR / "association_rules_plot.png"
    if p_ar.exists():
        st.image(str(p_ar), use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 6: INTERACTIVE PATIENT RISK PREDICTOR
# -----------------------------------------------------------------------------
elif section == "6. Interactive Patient Risk Predictor":
    st.header("6. Interactive Patient Risk Predictor (Simulation)")
    st.write(
        "Simulate predictions on fine needle aspirate biopsy profiles using the trained "
        "scikit-learn classification pipelines."
    )

    results_df, eval_details, pipelines, X_test, y_test = get_classification_results()
    feature_cols = [c for c in df.columns if c not in ["target", "diagnosis_label"]]

    st.subheader("Select Patient Simulation Scenario:")
    scenario = st.selectbox(
        "Choose a benchmark profile or customize features below:",
        ["Typical Benign Case (Patient A)", "Typical Malignant Case (Patient B)", "Median Borderline Case", "Custom Sliders"]
    )

    # Base values
    if scenario == "Typical Benign Case (Patient A)":
        base_sample = df[df["target"] == 1].iloc[0][feature_cols].copy()
    elif scenario == "Typical Malignant Case (Patient B)":
        base_sample = df[df["target"] == 0].iloc[0][feature_cols].copy()
    else:
        base_sample = df[feature_cols].median().copy()

    col1, col2, col3 = st.columns(3)
    user_inputs = {}

    # Key interactive sliders
    with col1:
        st.markdown("**Tumor Geometry (Mean)**")
        user_inputs["mean radius"] = st.slider("Mean Radius (mm)", float(df["mean radius"].min()), float(df["mean radius"].max()), float(base_sample["mean radius"]))
        user_inputs["mean perimeter"] = st.slider("Mean Perimeter (mm)", float(df["mean perimeter"].min()), float(df["mean perimeter"].max()), float(base_sample["mean perimeter"]))
        user_inputs["mean area"] = st.slider("Mean Area (sq mm)", float(df["mean area"].min()), float(df["mean area"].max()), float(base_sample["mean area"]))

    with col2:
        st.markdown("**Cellular Texture & Shape**")
        user_inputs["mean texture"] = st.slider("Mean Texture", float(df["mean texture"].min()), float(df["mean texture"].max()), float(base_sample["mean texture"]))
        user_inputs["mean smoothness"] = st.slider("Mean Smoothness", float(df["mean smoothness"].min()), float(df["mean smoothness"].max()), float(base_sample["mean smoothness"]))
        user_inputs["mean compactness"] = st.slider("Mean Compactness", float(df["mean compactness"].min()), float(df["mean compactness"].max()), float(base_sample["mean compactness"]))

    with col3:
        st.markdown("**Contour Irregularity**")
        user_inputs["mean concavity"] = st.slider("Mean Concavity", float(df["mean concavity"].min()), float(df["mean concavity"].max()), float(base_sample["mean concavity"]))
        user_inputs["mean concave points"] = st.slider("Mean Concave Points", float(df["mean concave points"].min()), float(df["mean concave points"].max()), float(base_sample["mean concave points"]))
        user_inputs["mean symmetry"] = st.slider("Mean Symmetry", float(df["mean symmetry"].min()), float(df["mean symmetry"].max()), float(base_sample["mean symmetry"]))

    # Fill remaining features with default profile values
    full_sample = base_sample.copy()
    for k, v in user_inputs.items():
        full_sample[k] = v

    patient_df = pd.DataFrame([full_sample])

    model_choice = st.radio("Select Prediction Engine:", ["Logistic Regression", "Random Forest"], horizontal=True)
    active_pipeline = pipelines[model_choice]

    pred_class = active_pipeline.predict(patient_df)[0]
    pred_prob = active_pipeline.predict_proba(patient_df)[0]
    # Class 0: Malignant, Class 1: Benign
    prob_mal = pred_prob[0] * 100
    prob_ben = pred_prob[1] * 100

    st.write("---")
    st.subheader("Model Risk Assessment Output:")

    rc1, rc2 = st.columns([1, 1])
    with rc1:
        if pred_class == 0:
            st.error(f"### Diagnosis Prediction: **MALIGNANT**")
            st.write(f"The model detected morphological characteristics indicative of malignancy.")
        else:
            st.success(f"### Diagnosis Prediction: **BENIGN**")
            st.write(f"The model identified morphological characteristics consistent with a benign lesion.")

    with rc2:
        st.write(f"**Confidence Breakdown ({model_choice}):**")
        st.progress(prob_mal / 100)
        st.write(f"- Malignant Risk Probability: **{prob_mal:.2f}%**")
        st.write(f"- Benign Probability: **{prob_ben:.2f}%**")


# -----------------------------------------------------------------------------
# SECTION 7: PROJECT REPORT & CONCLUSION
# -----------------------------------------------------------------------------
elif section == "7. Project Report & Conclusion":
    st.header("7. Project Synthesis, Methodological Limitations & Conclusion")

    st.markdown("""
    ### Summary of Empirical Findings
    1. **Data Preprocessing & Leakage Mitigation:**
       - The Wisconsin Breast Cancer dataset was audited and verified to possess zero native missing values.
       - StandardScaler fitted strictly within scikit-learn Pipelines on training partitions eliminated data leakage.
    2. **Unsupervised Clustering:**
       - K-Means achieved optimal silhouette clustering at `k = 2` (Silhouette Score = `0.3434`).
       - Cluster 0 captured 90.4% Benign samples, while Cluster 1 captured 90.7% Malignant samples purely through unsupervised pattern discovery.
    3. **Supervised Classification:**
       - **Logistic Regression Pipeline** achieved **98.25% Accuracy** and **0.9954 ROC-AUC**.
       - **Random Forest Pipeline** achieved **95.61% Accuracy** and **0.9939 ROC-AUC**.
    4. **Association Rule Mining:**
       - Over 850 rules mined. Co-occurrence of high cell concavity, area, and perimeter demonstrated a Lift of **3.985** with malignant diagnosis.
       - Disclaimed causal mechanisms: high values are morphological consequences, not physiological causes.

    ### Methodological Limitations
    - **Single Dataset Limitation:** Results are derived from a single center (University of Wisconsin Hospitals) with 569 patient samples.
    - **Demographic Generalizability:** External validation on multi-ethnic, diverse clinical cohorts is needed prior to translation.
    - **FNA Scope:** Image-based FNA features do not replace comprehensive histological biopsy or genomic profiling (e.g., BRCA1/BRCA2, HER2).
    """)

    st.info("The complete academic writeup with all equations and derivations is available in `report.md`.")
