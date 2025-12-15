"""
ModelAdapter - Adapter Pattern Implementation
Converts various ML model interfaces into a uniform interface
"""

import numpy as np
from typing import List, Union


class ModelAdapter:
    """
    Adapter Pattern: Provides a consistent interface for different ML model types
    (sklearn, custom, rule-based) to ensure uniform interaction across the application.
    
    Responsibilities:
    - Provide consistent predict() API
    - Provide consistent feature_names API
    - Handle predict_proba / predict differences
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Initialize the adapter with a model
        
        Args:
            model: Any ML model object (sklearn, custom, etc.)
            feature_names: List of feature names
        """
        self.model = model
        self._feature_names = feature_names
        
    def get_feature_names(self) -> List[str]:
        """
        Get feature names in a consistent format
        
        Returns:
            List of feature names
        """
        if self._feature_names:
            return self._feature_names
        
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'feature_names'):
            return list(self.model.feature_names)
        elif hasattr(self.model, 'n_features_in_'):
            return [f"Feature_{i}" for i in range(self.model.n_features_in_)]
        else:
            return [f"Feature_{i}" for i in range(10)]
    
    def predict(self, instance: Union[np.ndarray, List]) -> tuple:
        """
        Predict using the model with consistent output format
        
        Args:
            instance: Input instance (can be list or numpy array)
            
        Returns:
            Tuple of (prediction, probabilities if available)
        """
        if isinstance(instance, list):
            instance = np.array(instance).reshape(1, -1)
        elif isinstance(instance, np.ndarray) and instance.ndim == 1:
            instance = instance.reshape(1, -1)
            
        prediction = self.model.predict(instance)
        
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(instance)
        
        return prediction[0], probabilities[0] if probabilities is not None else None
    
    def get_model(self):
        """
        Get the underlying model object
        
        Returns:
            The wrapped model object
        """
        return self.model
    
    def get_model_type(self) -> str:
        """
        Get the type/name of the model
        
        Returns:
            String representing the model type
        """
        if hasattr(self.model, '__class__'):
            return self.model.__class__.__name__
        return str(type(self.model).__name__)
    
    def get_model_type(self) -> str:
        """
        Get the type/name of the model
        
        Returns:
            String representing the model type
        """
        if hasattr(self.model, '__class__'):
            return self.model.__class__.__name__
        return str(type(self.model).__name__)
