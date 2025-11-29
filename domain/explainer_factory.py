# domain/explainer_factory.py
from typing import List, Optional, Any

from .explainer_strategies import ExplainerBase, DecisionPathExplainer, SHAPExplainer
from .model_adapter import ModelAdapter # Required for type hinting/composition
import numpy as np

class ExplainerFactory:
    """
    Purpose: Creates the correct Explainer Strategy based on the model (Factory Method).
    Adheres to the Open/Closed Principle (OCP).
    """
    @staticmethod
    def create(model_adapter: ModelAdapter, background_data: Optional[np.ndarray] = None) -> ExplainerBase:
        
        model = model_adapter.model
        feature_names = model_adapter.get_feature_names()
        
        # OCP: Check if model is a tree and use the most specific explainer
        if hasattr(model, 'tree_'):
            print("Factory: Creating DecisionPathExplainer (Specific Strategy).")
            return DecisionPathExplainer(model_adapter, feature_names)
        
        # Check if we have background data and default to SHAP for complex models
        if background_data is not None:
            print("Factory: Creating SHAPExplainer (General Strategy).")
            return SHAPExplainer(model_adapter, feature_names, background_data)
            
        # Fallback (you can implement LIME here if needed)
        # For simplicity, we currently rely on the DecisionPath or SHAP logic.
        raise ValueError("Cannot create an Explainer. Model type not supported by existing strategies or background data is missing.")