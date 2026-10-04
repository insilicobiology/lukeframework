"""
Uncertainty estimation module for LUKE.
Directly interfaces with the native UBiQTree codebase.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Import the authentic UbiqTree class from the vendor module
try:
    from lukeframework.UbiqTree.UBiQTree.ubiqtree import UbiqTree
except ImportError:
    try:
        from .UbiqTree.UBiQTree.ubiqtree import UbiqTree
    except ImportError as e:
        raise ImportError(
            "Could not import UbiqTree from 'ubiqtree.py'. Please check the exact "
            "file name and class definition inside your UBiQTree vendor folder."
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

        # Initialize the native UBiQTree engine using the correct class name 'UbiqTree'
        self.engine = UbiqTree(model=self.model, X_train=self.X_train)

    def compute_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes SHAP values and authentic epistemic uncertainty scores
        using the native UBiQTree implementation.
        """
        # Call the native calculation method provided by UBiQTree
        shap_values, uncertainty_scores = self.engine.calculate_uncertainty_shap(X)

        return np.array(shap_values), np.array(uncertainty_scores)