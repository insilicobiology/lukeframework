"""
Plotting module for LUKE.
Generates publication-ready visualizations combining feature attributions
with epistemic uncertainty bounds for tabular data.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_uncertainty_summary(
        shap_values: np.ndarray,
        uncertainty_scores: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
        save_path: Optional[str] = None
):
    """
    Generates a summary bar chart showing global feature importance (mean absolute SHAP)
    paired with epistemic uncertainty error bounds.

    Args:
        shap_values: Array of SHAP values of shape (n_samples, n_features).
        uncertainty_scores: Array of epistemic uncertainty scores of shape (n_samples,) or (n_samples, n_features).
        feature_names: List of strings representing feature/biomarker names.
        top_n: Number of top features to display.
        save_path: Optional file path (e.g., 'seer_summary.png') to save the figure.
    """
    # Set up publication-style aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6))

    # Calculate mean absolute SHAP value for global feature importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Calculate average uncertainty impact per feature (or map global sample uncertainty)
    if uncertainty_scores.ndim == 1:
        # Broadcast sample-level uncertainty across features for error representation
        feature_uncertainty = np.std(shap_values * uncertainty_scores[:, np.newaxis], axis=0)
    else:
        feature_uncertainty = np.mean(uncertainty_scores, axis=0)

    # Sort features by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    top_indices = sorted_indices[:top_n]

    sorted_features = [feature_names[i] for i in top_indices]
    sorted_importance = mean_abs_shap[top_indices]
    sorted_errors = feature_uncertainty[top_indices]

    # Create horizontal bar plot with error bars representing epistemic variance
    y_pos = np.arange(len(sorted_features))

    plt.barh(
        y_pos,
        sorted_importance,
        xerr=sorted_errors,
        align='center',
        color='#2b5c8f',
        edgecolor='black',
        capsize=5,
        alpha=0.85
    )

    plt.yticks(y_pos, sorted_features)
    plt.gca().invert_yaxis()  # Labels read top-to-bottom

    plt.xlabel("Mean Absolute SHAP Value (Impact on Model)")
    plt.title("LUKE Biomarker Importance & Epistemic Uncertainty Summary", fontsize=13, fontweight='bold', pad=15)

    # Add a descriptive note for researchers
    plt.figtext(
        0.15, -0.05,
        "Note: Error bars represent model epistemic uncertainty. Larger bars indicate features with higher prediction variance.",
        wrap=True, horizontalalignment='left', fontsize=9, style='italic', color='gray'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to {save_path}")

    plt.show()