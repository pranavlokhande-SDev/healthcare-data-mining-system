"""
========================================================================================
Healthcare Disease Prediction and Patient Data Analysis Using Data Mining Techniques
Module: association_rules.py
Description: Frequent itemset extraction and Association Rule Mining using Apriori.
             Discretizes continuous clinical measurements into categorical intervals,
             calculates Support, Confidence, and Lift, and highlights that association
             does NOT imply causation.
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
from mlxtend.frequent_patterns import apriori, association_rules

from data_processing import load_raw_data

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def discretize_patient_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects key clinical morphologic features and discretizes them into
    tertiles ('Low', 'Medium', 'High') for association rule mining.

    Returns:
        pd.DataFrame: One-hot encoded boolean DataFrame ready for Apriori.
    """
    key_features = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean symmetry",
    ]

    discretized = pd.DataFrame(index=df.index)

    # Discretize each continuous feature into Low, Medium, High bins using tertiles
    for col in key_features:
        clean_name = col.replace("mean ", "")
        discretized[f"{clean_name}"] = pd.qcut(
            df[col],
            q=3,
            labels=["Low", "Med", "High"],
            duplicates="drop"
        ).astype(str)

    # Add diagnosis label
    discretized["diagnosis"] = df["diagnosis_label"]

    # Convert to one-hot boolean representation
    encoded_df = pd.get_dummies(discretized, dtype=bool)
    return encoded_df, discretized


def mine_association_rules(
    encoded_df: pd.DataFrame,
    min_support: float = 0.15,
    min_confidence: float = 0.60,
    min_lift: float = 1.2
) -> tuple:
    """
    Executes Apriori to extract frequent itemsets and derives association rules
    meeting support, confidence, and lift thresholds.

    Returns:
        tuple: (frequent_itemsets, rules_df, disease_rules_df)
    """
    # 1. Apriori algorithm
    frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame()

    # 2. Association rules generation
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame()

    # Filter by minimum lift
    rules = rules[rules["lift"] >= min_lift].copy()

    # Create readable representations of antecedents and consequents
    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["rule_str"] = rules["antecedents_str"] + "  ===>  " + rules["consequents_str"]

    # Sort rules primarily by lift, then confidence
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)

    # Filter rules where the consequent directly involves diagnosis
    disease_rules = rules[
        rules["consequents_str"].str.contains("diagnosis_", case=False, na=False)
    ].copy().reset_index(drop=True)

    return frequent_itemsets, rules, disease_rules


def plot_association_rules_visual(rules_df: pd.DataFrame, save_path: Path = None):
    """
    Plots an analytical scatter plot of Support vs Confidence colored by Lift,
    and a horizontal bar chart of the top 8 rules by Lift.
    """
    if rules_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.set_theme(style="whitegrid")

    # 1. Scatter Plot
    scatter = ax1.scatter(
        rules_df["support"],
        rules_df["confidence"],
        c=rules_df["lift"],
        cmap="plasma",
        s=rules_df["lift"] * 45,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Lift", fontsize=11)
    ax1.set_title("Association Rules: Support vs Confidence (Color = Lift)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Support", fontsize=11)
    ax1.set_ylabel("Confidence", fontsize=11)

    # 2. Bar Chart of Top Rules by Lift
    top_rules = rules_df.head(8).copy()
    top_rules["short_rule"] = top_rules["rule_str"].apply(
        lambda r: (r[:45] + "...") if len(r) > 48 else r
    )

    sns.barplot(
        data=top_rules,
        x="lift",
        y="short_rule",
        hue="short_rule",
        palette="mako",
        legend=False,
        ax=ax2
    )
    ax2.set_title("Top Association Rules by Lift Metric", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Lift Value", fontsize=11)
    ax2.set_ylabel("Association Rule", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def print_causation_disclaimer():
    """
    Prints a prominent explanation of the distinction between association and causation.
    """
    disclaimer = """
********************************************************************************
[CRITICAL DATA MINING PRINCIPLE: ASSOCIATION DOES NOT IMPLY CAUSATION]
--------------------------------------------------------------------------------
1. In Apriori association rule mining, a high confidence or lift between an item
   (e.g., 'concavity_High') and a consequent (e.g., 'diagnosis_Malignant') signifies
   a strong STATISTICAL CO-OCCURRENCE in the historical records of this dataset.
2. It DOES NOT prove that large cell concavity CAUSES malignant tumor growth.
   Both measurements are phenotypic morphological symptoms resulting from underlying
   genetic and biochemical cellular abnormalities.
3. Healthcare professionals and data scientists must treat association rules as
   hypotheses-generating clues, NOT causal mechanisms or standalone diagnostic verdicts.
********************************************************************************
"""
    print(disclaimer)


def run_association_pipeline():
    """
    Executes the complete association rule mining pipeline and saves outputs.
    """
    print("\n" + "=" * 70)
    print("   HEALTHCARE DATA MINING: ASSOCIATION RULE MINING (APRIORI)")
    print("=" * 70)

    print("\n[Step 1/4] Loading dataset and discretizing clinical measurements...")
    df = load_raw_data()
    encoded_df, discretized_df = discretize_patient_data(df)
    print(f" -> Discretized {discretized_df.shape[1] - 1} continuous features into tertiles (Low/Med/High).")
    print(f" -> Generated {encoded_df.shape[1]} one-hot binary items for Apriori.")

    print("\n[Step 2/4] Mining frequent itemsets & deriving association rules...")
    freq_items, rules, disease_rules = mine_association_rules(
        encoded_df, min_support=0.15, min_confidence=0.60, min_lift=1.2
    )
    print(f" -> Frequent Itemsets Discovered: {len(freq_items)}")
    print(f" -> Total Qualified Rules (Lift >= 1.2, Conf >= 0.60): {len(rules)}")
    print(f" -> Diagnostic Specific Rules (Consequent = Diagnosis): {len(disease_rules)}")

    print("\n[Step 3/4] Top 5 Strongest Diagnostic Rules by Lift:")
    if not disease_rules.empty:
        display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
        print(disease_rules[display_cols].head(5).round(3).to_string(index=False))
    else:
        print(" -> No diagnostic rules met the strict filter; showing top general rules:")
        print(rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(5).round(3).to_string(index=False))

    print("\n[Step 4/4] Saving results and generating visualization...")
    rules_csv_path = OUTPUTS_DIR / "association_rules.csv"
    plot_path = OUTPUTS_DIR / "association_rules_plot.png"

    # Export clean CSV
    export_df = rules[[
        "antecedents_str", "consequents_str", "support", "confidence", "lift",
        "leverage", "conviction"
    ]].copy().rename(columns={
        "antecedents_str": "antecedent",
        "consequents_str": "consequent"
    })
    export_df.to_csv(rules_csv_path, index=False)
    plot_association_rules_visual(rules, save_path=plot_path)

    print(f" -> Association Rules exported to: {rules_csv_path}")
    print(f" -> Association Visual saved to:   {plot_path}")

    # Display causation caveat
    print_causation_disclaimer()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_association_pipeline()
