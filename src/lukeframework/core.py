"""
Core orchestration classes for LUKE (Leverage Ubiqtree Keep Explanations).
Designed for generalizable tabular data analysis with tree-based ensembles.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Placeholder imports for underlying uncertainty and plotting modules
# (to be implemented in uncertainty.py and plotting.py)
from lukeframework.uncertainty import UBiQTreeUncertaintyWrapper
from lukeframework.plotting import plot_feature_uncertainty_summary


class LUKEPipeline:
    """
    End-to-end pipeline wrapper that combines standard tree ensemble training
    with UBiQTree epistemic uncertainty quantification.
    """

    def __init__(
            self,
            model: Union[BaseEstimator, ClassifierMixin, RegressorMixin],
            task: str = "classification"
    ):
        self.model = model
        self.task = task.lower()
        self.is_fitted = False
        self.uncertainty_engine: Optional[UBiQTreeUncertaintyWrapper] = None

        if self.task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "LUKEPipeline":
        """
        Fit the underlying tree model and initialize the UBiQTree uncertainty wrapper.

        Args:
            X: Feature dataframe (tabular data).
            y: Target labels or values.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        print(f"Fitting model for {self.task} task...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Initialize the underlying uncertainty engine using the trained model and training data
        self.uncertainty_engine = UBiQTreeUncertaintyWrapper(
            model=self.model,
            X_train=X,
            task=self.task
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate standard model predictions."""
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.model.predict(X)

    def evaluate_uncertainty(self, X: pd.DataFrame) -> "LUKEAnalyzer":
        """
        Compute epistemic uncertainty alongside SHAP attributions using UBiQTree.

        Args:
            X: Unseen or evaluation feature dataframe.

        Returns:
            A LUKEAnalyzer instance containing explanations and uncertainty metrics.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        print("Extracting SHAP values and quantifying epistemic uncertainty...")
        shap_values, uncertainty_scores = self.uncertainty_engine.compute_uncertainty(X)

        return LUKEAnalyzer(
            X=X,
            shap_values=shap_values,
            uncertainty_scores=uncertainty_scores,
            feature_names=list(X.columns)
        )

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted using .fit() before evaluating.")


class LUKEAnalyzer:
    """
    Post-hoc analysis container for managing explanations, filtering low-confidence
    predictions, and generating publication-ready visuals.
    """

    def __init__(
            self,
            X: pd.DataFrame,
            shap_values: np.ndarray,
            uncertainty_scores: np.ndarray,
            feature_names: list
    ):
        self.X = X
        self.shap_values = shap_values
        self.uncertainty_scores = uncertainty_scores
        self.feature_names = feature_names

    def filter_by_confidence(self, threshold: float = 0.80) -> "LUKEAnalyzer":
        """
        Filter out samples where epistemic uncertainty exceeds acceptable tolerances,
        retaining only high-confidence predictions.

        Args:
            threshold: Percentile or raw threshold score for model confidence.
        """
        # Example logic: keep rows where uncertainty is below the given quantile threshold
        cutoff = np.quantile(self.uncertainty_scores, threshold)
        mask = self.uncertainty_scores <= cutoff

        filtered_analyzer = LUKEAnalyzer(
            X=self.X.iloc[mask].reset_index(drop=True),
            shap_values=self.shap_values[mask],
            uncertainty_scores=self.uncertainty_scores[mask],
            feature_names=self.feature_names
        )
        print(f"Retained {mask.sum()} out of {len(mask)} samples after confidence filtering.")
        return filtered_analyzer

    def plot_biomarker_confidence(self, save_path: Optional[str] = None):
        """
        Generate a clean summary plot displaying feature attributions
        coupled with epistemic uncertainty error bounds.
        """
        plot_feature_uncertainty_summary(
            shap_values=self.shap_values,
            uncertainty_scores=self.uncertainty_scores,
            feature_names=self.feature_names,
            save_path=save_path
        )