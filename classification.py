"""
========================================================================================
Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques
Module: classification.py
Description: Supervised classification using Logistic Regression and Random Forest.
             Utilizes scikit-learn Pipelines to strictly prevent data leakage.
             Evaluates Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC.
========================================================================================
NOTE & MEDICAL DISCLAIMER:
This project uses the Breast Cancer Wisconsin educational dataset from scikit-learn.
It is an educational project and NOT a medical diagnostic tool.
========================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from data_processing import load_raw_data

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_pipelines():
    """
    Constructs scikit-learn Pipelines for Logistic Regression and Random Forest.
    Scaling is included inside the Pipeline so it is fitted strictly on the
    training fold, preventing data leakage.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    return pipelines


def train_and_evaluate_models():
    """
    Splits data (80:20 stratified), trains models via pipelines, and computes
    comprehensive evaluation metrics.

    Returns:
        tuple: (results_df, eval_details, trained_pipelines, X_test, y_test)
    """
    df = load_raw_data()
    feature_cols = [c for c in df.columns if c not in ["target", "diagnosis_label"]]
    X = df[feature_cols]
    y = df["target"]

    # Stratified 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipelines = build_pipelines()
    results = []
    eval_details = {}

    for name, pipeline in pipelines.items():
        # Fit pipeline strictly on training data
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of Benign (1)

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        # Note: Class 0 is Malignant, Class 1 is Benign
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec_macro, 4),
            "Recall (Macro)": round(rec_macro, 4),
            "F1-Score (Macro)": round(f1_macro, 4),
            "ROC-AUC": round(roc_auc, 4)
        })

        eval_details[name] = {
            "confusion_matrix": cm,
            "classification_report": cr,
            "y_prob": y_prob,
            "y_pred": y_pred
        }

    results_df = pd.DataFrame(results)
    return results_df, eval_details, pipelines, X_test, y_test


def plot_confusion_matrices(eval_details: dict, save_path: Path = None):
    """
    Plots side-by-side Confusion Matrix heatmaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.set_theme(style="white")

    classes = ["Malignant (0)", "Benign (1)"]

    for idx, (name, details) in enumerate(eval_details.items()):
        cm = details["confusion_matrix"]
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else "Greens",
            cbar=False,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            annot_kws={"size": 14, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Diagnosis", fontsize=11)
        ax.set_ylabel("True Diagnosis", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(eval_details: dict, y_test: pd.Series, save_path: Path = None):
    """
    Plots ROC curves for both models with AUC annotation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c"}

    for name, details in eval_details.items():
        fpr, tpr, _ = roc_curve(y_test, details["y_prob"])
        auc_val = roc_auc_score(y_test, details["y_prob"])
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_val:.4f})",
            color=colors.get(name, "#333333"),
            linewidth=2.5
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save_path: Path = None):
    """
    Generates a grouped bar chart comparing key evaluation metrics between models.
    """
    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "ROC-AUC"]
    melted_df = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#1f77b4", "#2ca02c"],
        ax=ax
    )

    ax.set_title("Classification Model Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.legend(title="Model", frameon=True)

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(
                f"{val:.3f}",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def run_classification_pipeline():
    """
    Executes the complete classification pipeline and logs evaluation results.
    """
    print("\n" + "=" * 70)
    print("   HEALTHCARE DATA MINING: SUPERVISED CLASSIFICATION")
    print("=" * 70)

    print("\n[Step 1/4] Training Logistic Regression & Random Forest via Pipelines...")
    results_df, eval_details, pipelines, X_test, y_test = train_and_evaluate_models()

    print("\n[Step 2/4] Model Performance Comparison Table (Test Set N=114):")
    print(results_df.to_string(index=False))

    print("\n[Step 3/4] Detailed Classification Reports:")
    for name, details in eval_details.items():
        print(f"\n--- {name} Classification Report ---")
        cr_df = pd.DataFrame(details["classification_report"]).transpose()
        print(cr_df.round(4).to_string())

    print("\n[Step 4/4] Generating and saving evaluation charts & metrics...")
    cm_plot_path = OUTPUTS_DIR / "confusion_matrices.png"
    roc_plot_path = OUTPUTS_DIR / "roc_curves.png"
    comp_plot_path = OUTPUTS_DIR / "model_comparison.png"
    results_csv_path = OUTPUTS_DIR / "model_comparison.csv"

    plot_confusion_matrices(eval_details, save_path=cm_plot_path)
    plot_roc_curves(eval_details, y_test, save_path=roc_plot_path)
    plot_model_comparison(results_df, save_path=comp_plot_path)
    results_df.to_csv(results_csv_path, index=False)

    print(f" -> Model Comparison CSV saved to:      {results_csv_path}")
    print(f" -> Confusion Matrices Plot saved to:   {cm_plot_path}")
    print(f" -> ROC Curves Plot saved to:           {roc_plot_path}")
    print(f" -> Model Comparison Chart saved to:    {comp_plot_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_classification_pipeline()
