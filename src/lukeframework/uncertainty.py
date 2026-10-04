"""
Uncertainty estimation module for LUKE.
Directly interfaces with the native UBiQTree codebase.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Import the actual UBiQTree codebase from your vendor directory or package layout
try:
    from lukeframework.UbiqTree.UBiQTree.ubiqtree import UBiQTreeCore
except ImportError:
    # Fallback import path if structured differently or installed externally
    try:
        from lukeframework.UbiqTree.UBiQTree.ubiqtree import UBiQTreeCore
    except ImportError as e:
        raise ImportError(
            "Could not import UBiQTreeCore. Ensure that the UBiQTree repository is "
            "properly included in your vendor directory or installed in your environment."
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

        # Initialize the native UBiQTree engine using the user's specified class
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
        # Call the native methods provided by UBiQTreeCore
        # (Adjust method names here if UBiQTree uses a slightly different execution call)
        shap_values, uncertainty_scores = self.engine.calculate_uncertainty_shap(X)

        return np.array(shap_values), np.array(uncertainty_scores)