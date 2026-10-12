"""
========================================================================================
Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques
Module: clustering.py
Description: Unsupervised clustering using K-Means, Elbow Method, Silhouette Analysis,
             PCA 2D projection, and cluster profiling.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Import loading function from data_processing module
from data_processing import load_raw_data

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_kmeans_range(X_scaled: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """
    Evaluates K-Means clustering across a range of k values using:
    - Inertia (Sum of squared distances to cluster center) for Elbow Method
    - Silhouette Score for cluster cohesion and separation

    Returns:
        pd.DataFrame: Table with columns ['k', 'inertia', 'silhouette_score']
    """
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": round(kmeans.inertia_, 2),
            "silhouette_score": round(sil_score, 4)
        })
    return pd.DataFrame(results)


def perform_pca_clustering(df: pd.DataFrame, n_clusters: int = 2):
    """
    Standardizes features, performs PCA to 2 components, and fits K-Means clustering.

    Returns:
        tuple: (pca_df, kmeans_model, pca_model, metrics_df, profile_df)
    """
    feature_cols = [c for c in df.columns if c not in ["target", "diagnosis_label"]]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Range evaluation
    metrics_df = evaluate_kmeans_range(X_scaled, k_range=range(2, 11))

    # 2. Optimal K-Means Model fit
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 3. PCA 2D Dimensionality Reduction
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Cluster": cluster_labels,
        "Cluster_Label": [f"Cluster {c}" for c in cluster_labels],
        "True_Diagnosis": df["diagnosis_label"].values,
        "Target": df["target"].values
    })

    # 4. Cluster Profiling on key clinical features
    key_features = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity", "mean symmetry"
    ]
    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = cluster_labels

    profile_df = df_with_clusters.groupby("Cluster")[key_features].mean().round(3)
    profile_df["Sample_Count"] = df_with_clusters.groupby("Cluster").size()

    # Determine alignment with diagnosis
    contingency = pd.crosstab(
        df_with_clusters["Cluster"],
        df_with_clusters["diagnosis_label"]
    )
    profile_df = profile_df.join(contingency)

    return pca_df, kmeans, pca, metrics_df, profile_df


def plot_elbow_and_silhouette(metrics_df: pd.DataFrame, save_path: Path = None):
    """
    Generates dual subplot: Elbow curve and Silhouette score vs k.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    # Elbow curve (Inertia)
    ax1.plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    ax1.set_title("K-Means Elbow Method (Inertia)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax1.set_xticks(metrics_df["k"])

    # Silhouette curve
    best_k = metrics_df.loc[metrics_df["silhouette_score"].idxmax()]["k"]
    ax2.plot(metrics_df["k"], metrics_df["silhouette_score"], marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
    ax2.axvline(best_k, color="#d62728", linestyle="--", alpha=0.8, label=f"Max Silhouette (k={int(best_k)})")
    ax2.set_title("Silhouette Score vs Number of Clusters", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_xticks(metrics_df["k"])
    ax2.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pca_clusters(pca_df: pd.DataFrame, pca_model: PCA, save_path: Path = None):
    """
    Generates side-by-side PCA scatter plots:
    Left: K-Means Discovered Clusters
    Right: True Clinical Diagnosis (Malignant vs Benign)
    """
    var_exp = pca_model.explained_variance_ratio_ * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.set_theme(style="whitegrid")

    # Plot K-Means Clusters
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster_Label",
        palette=["#1f77b4", "#ff7f0e"],
        alpha=0.85,
        s=55,
        edgecolor="w",
        linewidth=0.5,
        ax=ax1
    )
    ax1.set_title("Unsupervised K-Means Clusters in PCA Space", fontsize=13, fontweight="bold")
    ax1.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)", fontsize=11)
    ax1.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)", fontsize=11)
    ax1.legend(title="K-Means Cluster", frameon=True)

    # Plot True Diagnosis
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="True_Diagnosis",
        palette={"Malignant": "#d62728", "Benign": "#2ca02c"},
        alpha=0.85,
        s=55,
        edgecolor="w",
        linewidth=0.5,
        ax=ax2
    )
    ax2.set_title("True Biological Diagnosis in PCA Space", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)", fontsize=11)
    ax2.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)", fontsize=11)
    ax2.legend(title="True Diagnosis", frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def run_clustering_pipeline():
    """
    Executes the complete clustering pipeline and saves outputs.
    """
    print("\n" + "=" * 70)
    print("   HEALTHCARE DATA MINING: UNSUPERVISED CLUSTERING & PCA")
    print("=" * 70)

    print("\n[Step 1/4] Loading cleaned dataset...")
    df = load_raw_data()

    print("\n[Step 2/4] Evaluating K-Means across k=2 to k=10 (Elbow & Silhouette)...")
    pca_df, kmeans, pca, metrics_df, profile_df = perform_pca_clustering(df, n_clusters=2)
    print(metrics_df.to_string(index=False))

    best_k = int(metrics_df.loc[metrics_df["silhouette_score"].idxmax()]["k"])
    best_sil = metrics_df.loc[metrics_df["silhouette_score"].idxmax()]["silhouette_score"]
    print(f"\n -> Optimal Silhouette Score achieved at k={best_k} (Score: {best_sil})")

    print("\n[Step 3/4] Cluster Profiles (Mean Values of Clinical Features):")
    print(profile_df.to_string())

    print("\n[Step 4/4] Generating and saving clustering visualizations...")
    elbow_plot_path = OUTPUTS_DIR / "elbow_silhouette_plot.png"
    pca_plot_path = OUTPUTS_DIR / "kmeans_pca_clusters.png"
    metrics_path = OUTPUTS_DIR / "clustering_metrics.csv"
    profile_path = OUTPUTS_DIR / "cluster_profile.csv"

    plot_elbow_and_silhouette(metrics_df, save_path=elbow_plot_path)
    plot_pca_clusters(pca_df, pca, save_path=pca_plot_path)
    metrics_df.to_csv(metrics_path, index=False)
    profile_df.to_csv(profile_path)

    print(f" -> Elbow & Silhouette Plot saved to: {elbow_plot_path}")
    print(f" -> PCA Cluster Scatter Plot saved to: {pca_plot_path}")
    print(f" -> Cluster Metrics CSV saved to:      {metrics_path}")
    print(f" -> Cluster Profile CSV saved to:      {profile_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_clustering_pipeline()
