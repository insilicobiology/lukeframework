"""
Example Script: Analyzing SEER Breast Cancer Data using LUKE
(Leverage Ubiqtree Keep Explanations)

This script shows how a researcher can train a standard tree ensemble
on clinical tabular data and evaluate prediction uncertainty seamlessly.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Import LUKE components from your package
from src.lukeframework.core import LUKEPipeline


def load_synthetic_or_local_seer_data():
    """
    Helper to load the SEER breast cancer dataset.
    For demonstration purposes, if the local file isn't found,
    we generate a mock clinical dataframe mimicking SEER structure.
    """
    file_path = "examples/data/SEERBreastCancerDataset.csv"

    if os.path.exists(file_path):
        print(f"Loading local dataset from {file_path}...")
        df = pd.read_csv(file_path)
    else:
        print("Local SEER dataset file not found. Generating a synthetic clinical mock dataset for testing...")
        np.random.seed(42)
        n_samples = 1000

        df = pd.DataFrame({
            "Age": np.random.randint(30, 85, size=n_samples),
            "T_Stage": np.random.randint(1, 5, size=n_samples),
            "N_Stage": np.random.randint(1, 4, size=n_samples),
            "Grade": np.random.randint(1, 4, size=n_samples),
            "Tumor_Size": np.random.exponential(20, size=n_samples) + 5,
            "Estrogen_Receptor_Status": np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
            "Progesterone_Receptor_Status": np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
            "Survival_Status": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
        })

    return df


def main():
    # 1. Load Data
    data = load_synthetic_or_local_seer_data()

    # Separate features and target label
    # Assuming 'Survival_Status' or the last column is the binary target
    target_column = "Survival_Status" if "Survival_Status" in data.columns else data.columns[-1]

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training cohort shape: {X_train.shape}")
    print(f"Evaluation cohort shape: {X_test.shape}")

    # 3. Initialize Standard ML Model (e.g., XGBoost Classifier)
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    # 4. Wrap with LUKE Pipeline
    print("\n--- Initializing LUKE Pipeline ---")
    pipeline = LUKEPipeline(model=base_model, task="classification")

    # Fit the model and initialize the UBiQTree uncertainty engine
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Uncertainty on Patient Test Cohort
    print("\n--- Evaluating Epistemic Uncertainty with UBiQTree ---")
    analyzer = pipeline.evaluate_uncertainty(X_test)

    # 6. Keep Explanations Clean: Filter out high-uncertainty (unstable) predictions
    print("\n--- Filtering Cohort by Model Confidence ---")
    high_confidence_analyzer = analyzer.filter_by_confidence(threshold=0.80)

    # 7. Generate Publication-Ready Visual Summary
    print("\n--- Generating Publication-Ready Plot ---")
    output_plot_path = "seer_luke_uncertainty_summary.png"
    high_confidence_analyzer.plot_biomarker_confidence(save_path=output_plot_path)

    print(f"\nPipeline execution complete! Summary plot saved to '{output_plot_path}'.")


if __name__ == "__main__":
    main()