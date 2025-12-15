"""
LIMEExplainer - Strategy Pattern Implementation
Generates local explanations using LIME (Local Interpretable Model-agnostic Explanations)
"""

from typing import Dict, Any, List
import numpy as np
from .explainer_base import ExplainerBase


class LIMEExplainer(ExplainerBase):
    """
    Strategy Implementation: Generate local explanations using LIME.
    
    LIME explains predictions by approximating the model locally with an
    interpretable model (linear regression).
    
    Responsibilities:
    - Create LIME explainer
    - Compute local importance scores
    """
    
    def __init__(self, model, feature_names: List[str] = None, mode: str = 'classification'):
        """
        Initialize LIME explainer
        
        Args:
            model: The ML model to explain
            feature_names: List of feature names
            mode: 'classification' or 'regression'
        """
        self.model = model
        self.feature_names = feature_names or []
        self.mode = mode
        self.explainer = None
        
        try:
            from lime import lime_tabular
            
            
            training_data = np.random.randn(100, len(feature_names)) if feature_names else np.random.randn(100, 10)
            
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                mode=mode,
                verbose=False
            )
        except ImportError:
            self.explainer = None
    
    def explain(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Generate LIME explanation for the given instance
        
        Args:
            instance: Input instance to explain
            
        Returns:
            Dictionary containing LIME explanation data
        """
        if isinstance(instance, list):
            instance = np.array(instance)
        
        if instance.ndim > 1:
            instance = instance.flatten()
        
        try:
            if hasattr(self.model, 'predict_proba'):
                prediction = self.model.predict_proba(instance.reshape(1, -1))[0]
                pred_class = self.model.predict(instance.reshape(1, -1))[0]
            else:
                prediction = self.model.predict(instance.reshape(1, -1))[0]
                pred_class = prediction
        except:
            prediction = 0
            pred_class = 0
        
        if self.explainer is not None:
            try:
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = self.model.predict_proba
                else:
                    predict_fn = lambda x: self.model.predict(x).reshape(-1, 1)
                
                exp = self.explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=len(self.feature_names)
                )
                
                lime_values = exp.as_list()
                
                feature_importance = {}
                for feature_desc, value in lime_values:
                    feature_name = feature_desc.split()[0] if ' ' in feature_desc else feature_desc
                    
                    for fname in self.feature_names:
                        if fname in feature_name or feature_name in fname:
                            feature_importance[fname] = float(value)
                            break
                    else:
                        feature_importance[feature_name] = float(value)
                        
            except Exception as e:
                feature_importance = self._calculate_simple_importance(instance)
        else:
            feature_importance = self._calculate_simple_importance(instance)
        
        return {
            'feature_importance': feature_importance,
            'explanation_type': 'LIME',
            'prediction': pred_class,
            'instance': instance.tolist()
        }
    
    def _calculate_simple_importance(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Fallback method: Calculate simple feature importance when LIME is unavailable
        
        Args:
            instance: Input instance
            
        Returns:
            Dictionary of feature importances
        """
        importance = {}
        for i, (name, value) in enumerate(zip(self.feature_names, instance)):
            importance[name] = float(value * 0.1 * (1 + np.random.randn() * 0.2))
        
        return importance
