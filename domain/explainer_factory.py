"""
ExplainerFactory - Factory Method Pattern Implementation
Automatically selects and creates the best explainer based on model type
"""

from typing import Optional, List
import numpy as np
from .explainer_base import ExplainerBase
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .decision_path_explainer import DecisionPathExplainer


class ExplainerFactory:
    """
    Factory Method Pattern: Returns the best possible explainer automatically.
    
    This factory inspects the model type and automatically selects the most
    appropriate explanation algorithm (SHAP, LIME, or DecisionPath).
    
    Responsibilities:
    - Inspect model type
    - Provide SHAP, LIME, or DecisionPath explainer
    - Handle explainer instantiation
    """
    
    @staticmethod
    def create(
        model,
        background: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        explainer_type: Optional[str] = None
    ) -> ExplainerBase:
        """
        Create the appropriate explainer based on model type or specified type
        
        Args:
            model: The ML model to explain
            background: Background data for SHAP (optional)
            feature_names: List of feature names
            explainer_type: Force specific explainer ('shap', 'lime', 'decision_path')
                           If None, automatically select based on model
        
        Returns:
            ExplainerBase instance (SHAP, LIME, or DecisionPath explainer)
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(10)]
        
        if explainer_type:
            return ExplainerFactory._create_specific_explainer(
                explainer_type, model, background, feature_names
            )
        
        model_class = model.__class__.__name__
        
        if any(tree_type in model_class for tree_type in 
               ['DecisionTree', 'RandomForest', 'ExtraTree', 'GradientBoosting']):
            print(f"Detected tree-based model: {model_class}. Using DecisionPath explainer.")
            return DecisionPathExplainer(model, feature_names)
        
        try:
            import shap
            print(f"Using SHAP explainer for model: {model_class}")
            return SHAPExplainer(model, background, feature_names)
        except ImportError:
            try:
                import lime
                print(f"SHAP not available. Using LIME explainer for model: {model_class}")
                
                mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
                return LIMEExplainer(model, feature_names, mode)
            except ImportError:
                print(f"Neither SHAP nor LIME available. Using DecisionPath explainer.")
                return DecisionPathExplainer(model, feature_names)
    
    @staticmethod
    def _create_specific_explainer(
        explainer_type: str,
        model,
        background: Optional[np.ndarray],
        feature_names: List[str]
    ) -> ExplainerBase:
        """
        Create a specific type of explainer
        
        Args:
            explainer_type: Type of explainer ('shap', 'lime', 'decision_path')
            model: The ML model
            background: Background data
            feature_names: Feature names
            
        Returns:
            Requested explainer instance
        """
        explainer_type = explainer_type.lower()
        
        if explainer_type == 'shap':
            return SHAPExplainer(model, background, feature_names)
        
        elif explainer_type == 'lime':
            mode = 'classification' if hasattr(model, 'predict_proba') else 'regression'
            return LIMEExplainer(model, feature_names, mode)
        
        elif explainer_type in ['decision_path', 'decisionpath', 'tree']:
            return DecisionPathExplainer(model, feature_names)
        
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}. "
                           f"Valid types: 'shap', 'lime', 'decision_path'")
    
    @staticmethod
    def get_available_explainers() -> List[str]:
        """
        Get list of available explainer types based on installed packages
        
        Returns:
            List of available explainer names
        """
        available = ['decision_path'] 
        
        try:
            import shap
            available.append('shap')
        except ImportError:
            pass
        
        try:
            import lime
            available.append('lime')
        except ImportError:
            pass
        
        return available
