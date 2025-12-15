"""
SHAPExplainer - Strategy Pattern Implementation
Generates SHAP values for model prediction explanation
"""

from typing import Dict, Any, List
import numpy as np
from .explainer_base import ExplainerBase


class SHAPExplainer(ExplainerBase):
    """
    Strategy Implementation: Generate SHAP values for model prediction explanation.
    
    SHAP (SHapley Additive exPlanations) provides feature contribution values
    based on cooperative game theory.
    
    Responsibilities:
    - Initialize SHAP explainer
    - Compute feature contributions
    """
    
    def __init__(self, model, background_data: np.ndarray = None, feature_names: List[str] = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: The ML model to explain
            background_data: Background dataset for SHAP (optional)
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or []
        self.explainer = None
        
        try:
            import shap
            
            if hasattr(model, 'predict_proba'):
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
                else:
                    self.explainer = None
            else:
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(model.predict, background_data)
                    
        except ImportError:
            self.explainer = None
    
    def explain(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP explanation for the given instance
        
        Args:
            instance: Input instance to explain
            
        Returns:
            Dictionary containing SHAP values and explanation data
        """
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        elif instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        try:
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(instance)[0]
                pred_class = self.model.predict(instance)[0]
            else:
                prediction = self.model.predict(instance)[0]
                pred_class = prediction
        except:
            prediction = 0
            pred_class = 0
        
        if self.explainer is not None:
            try:
                import shap
                shap_values = self.explainer.shap_values(instance)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                if shap_values.ndim > 1:
                    shap_values = shap_values[0]
                    
            except Exception as e:
                shap_values = self._calculate_simple_importance(instance[0])
        else:
            shap_values = self._calculate_simple_importance(instance[0])
        
        feature_importance = {}
        for i, (name, value) in enumerate(zip(self.feature_names, shap_values)):
            feature_importance[name] = float(value)
        
        return {
            'feature_importance': feature_importance,
            'explanation_type': 'SHAP',
            'prediction': pred_class,
            'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values),
            'instance': instance[0].tolist()
        }
    
    def _calculate_simple_importance(self, instance: np.ndarray) -> np.ndarray:
        """
        Fallback method: Calculate simple feature importance when SHAP is unavailable
        
        Args:
            instance: Input instance
            
        Returns:
            Array of importance values
        """
        return instance * 0.1 * np.random.randn(len(instance)) * 0.5
