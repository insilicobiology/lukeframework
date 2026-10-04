"""
Uncertainty estimation module for LUKE.
Directly interfaces with the native UBiQTree codebase.
"""

from typing import Tuple
import numpy as np
import pandas as pd

# Safe import from the ubiqtree module file inside the package structure
try:
    from lukeframework.UbiqTree.UBiQTree.ubiqtree import UBiQTreeCore
except ImportError:
    try:
        # Fallback to alternate casing or relative import
        from .UbiqTree.UBiQTree.ubiqtree import UBiQTreeCore
    except ImportError:
        try:
            # If the class name itself is just UBiQTree or Ubiqtree
            from lukeframework.UbiqTree.UBiQTree.ubiqtree import UBiQTree as UBiQTreeCore
        except ImportError as e:
            raise ImportError(
                "Could not import UBiQTreeCore from 'ubiqtree.py'. Please check the class name "
                "defined inside your vendor 'ubiqtree.py' file."
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
        """
        shap_values, uncertainty_scores = self.engine.calculate_uncertainty_shap(X)

        return np.array(shap_values), np.array(uncertainty_scores)