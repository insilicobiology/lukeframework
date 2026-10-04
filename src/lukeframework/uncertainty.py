"""
Uncertainty estimation module for LUKE.
Directly interfaces with the native UBiQTree codebase.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Safe import cascade for nested UBiQTree repository layout
try:
    from .UbiqTree.UBiQTree import UBiQTreeCore
except ImportError:
    try:
        from lukeframework.UbiqTree.UBiQTree import UBiQTreeCore
    except ImportError as e:
        raise ImportError(
            "Could not import UBiQTreeCore. Ensure that the UBiQTree repository is placed "
            "inside 'src/lukeframework/UbiqTree/' and that all nested folders contain an '__init__.py' file."
        ) from e


class UBiQTreeUncertaintyWrapper:
    """
    Bridge wrapper that invokes the genuine UBiQTree backend
    to compute epistemic uncertainty in tree ensemble SHAP values.
    """

    def __init__(self, model: object, X_train: pd.DataFrame, task: str = "classification"):
        self.model = model
        self.X_train = X_train
        self.task = task

        # Initialize the native UBiQTree engine
        self.engine = UBiQTreeCore(model=self.model, X_train=self.X_train)

    def compute_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes SHAP values and authentic epistemic uncertainty scores
        using the native UBiQTree implementation.

        Args:
            X: Evaluation feature DataFrame (e.g., patient cohort).

        Returns:
            A tuple of (shap_values, epistemic_uncertainty_scores).
        """
        # Call the native method from UBiQTreeCore
        shap_values, uncertainty_scores = self.engine.calculate_uncertainty_shap(X)

        return np.array(shap_values), np.array(uncertainty_scores)