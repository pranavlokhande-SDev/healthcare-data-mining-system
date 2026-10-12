"""
========================================================================================
Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques
Module: visualization.py
Description: Master visualization engine generating all exploratory, clustering,
             classification, and association rule plots saved to outputs/.
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

from data_processing import load_raw_data
from clustering import perform_pca_clustering, plot_elbow_and_silhouette, plot_pca_clusters
from classification import (
    train_and_evaluate_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison
)
from association_rules import (
    discretize_patient_data,
    mine_association_rules,
    plot_association_rules_visual
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, save_path: Path = None):
    """
    Generates a publication-quality bar chart of the diagnostic class distribution.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_theme(style="whitegrid")

    palette = {"Malignant": "#d62728", "Benign": "#2ca02c"}
    counts = df["diagnosis_label"].value_counts()
    total = len(df)

    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=palette,
        legend=False,
        ax=ax,
        edgecolor="black",
        linewidth=1.2
    )

    ax.set_title("Breast Cancer Wisconsin: Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Clinical Diagnosis", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)

    # Bar annotations with count and percentage
    for i, (label, count) in enumerate(counts.items()):
        pct = (count / total) * 100
        ax.text(
            i,
            count + 10,
            f"{count} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = None):
    """
    Generates a correlation heatmap across key primary 'mean' morphologic features.
    """
    mean_cols = [c for c in df.columns if c.startswith("mean ")]
    corr = df[mean_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_theme(style="white")

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-0.5,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )

    clean_labels = [c.replace("mean ", "").capitalize() for c in mean_cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(clean_labels, rotation=0, fontsize=10)
    ax.set_title("Feature Correlation Heatmap (Mean Morphological Indicators)", fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, save_path: Path = None):
    """
    Plots boxplots comparing the distribution of 4 primary clinical features
    between Malignant and Benign diagnoses.
    """
    features_to_plot = ["mean radius", "mean texture", "mean perimeter", "mean area"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.set_theme(style="whitegrid")

    palette = {"Malignant": "#d62728", "Benign": "#2ca02c"}

    for idx, col in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(
            data=df,
            x="diagnosis_label",
            y=col,
            hue="diagnosis_label",
            palette=palette,
            legend=False,
            ax=ax,
            width=0.45
        )
        clean_name = col.replace("mean ", "").capitalize()
        ax.set_title(f"Distribution of Mean {clean_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Diagnosis", fontsize=10)
        ax.set_ylabel(f"{clean_name} Value", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def generate_all_visualizations():
    """
    Executes and saves all 9 visual plots required for the project.
    """
    print("\n" + "=" * 70)
    print("   HEALTHCARE DATA MINING: MASTER VISUALIZATION GENERATOR")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading patient dataset...")
    df = load_raw_data()

    # 2. Exploratory Plots
    print("\n[2/5] Generating Exploratory Visualizations...")
    p1 = OUTPUTS_DIR / "class_distribution.png"
    p2 = OUTPUTS_DIR / "correlation_heatmap.png"
    p3 = OUTPUTS_DIR / "feature_distributions.png"

    plot_class_distribution(df, save_path=p1)
    print(f" -> Saved Class Distribution:      {p1.name}")

    plot_correlation_heatmap(df, save_path=p2)
    print(f" -> Saved Correlation Heatmap:     {p2.name}")

    plot_feature_distributions(df, save_path=p3)
    print(f" -> Saved Feature Distributions:   {p3.name}")

    # 3. Clustering Visualizations
    print("\n[3/5] Generating Clustering Visualizations...")
    pca_df, kmeans, pca, metrics_df, profile_df = perform_pca_clustering(df, n_clusters=2)
    p4 = OUTPUTS_DIR / "elbow_silhouette_plot.png"
    p5 = OUTPUTS_DIR / "kmeans_pca_clusters.png"

    plot_elbow_and_silhouette(metrics_df, save_path=p4)
    print(f" -> Saved Elbow & Silhouette:      {p4.name}")

    plot_pca_clusters(pca_df, pca, save_path=p5)
    print(f" -> Saved PCA Cluster Scatter:     {p5.name}")

    # 4. Classification Visualizations
    print("\n[4/5] Generating Classification Visualizations...")
    results_df, eval_details, pipelines, X_test, y_test = train_and_evaluate_models()
    p6 = OUTPUTS_DIR / "confusion_matrices.png"
    p7 = OUTPUTS_DIR / "roc_curves.png"
    p8 = OUTPUTS_DIR / "model_comparison.png"

    plot_confusion_matrices(eval_details, save_path=p6)
    print(f" -> Saved Confusion Matrices:      {p6.name}")

    plot_roc_curves(eval_details, y_test, save_path=p7)
    print(f" -> Saved ROC Curves:              {p7.name}")

    plot_model_comparison(results_df, save_path=p8)
    print(f" -> Saved Model Comparison Chart:  {p8.name}")

    # 5. Association Rule Mining Visualizations
    print("\n[5/5] Generating Association Rule Visualizations...")
    encoded_df, _ = discretize_patient_data(df)
    _, rules, _ = mine_association_rules(encoded_df)
    p9 = OUTPUTS_DIR / "association_rules_plot.png"

    plot_association_rules_visual(rules, save_path=p9)
    print(f" -> Saved Association Rules Plot:  {p9.name}")

    print("\n" + "=" * 70)
    print(f" [SUCCESS] All 9 high-resolution visualizations saved under:")
    print(f"           {OUTPUTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_all_visualizations()
