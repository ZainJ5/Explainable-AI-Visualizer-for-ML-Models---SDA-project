# domain/model_adapter.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any

# --- 1. Abstract Adapter Interface (Dependency Inversion Principle) ---

class ModelAdapter(ABC):
    """
    Purpose: Convert any ML model into a uniform interface. [cite: 37]
    Defines a consistent API for prediction and feature metadata.
    """
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Provides consistent feature_names [cite: 40, 43]"""
        pass

    @abstractmethod
    def predict(self, instance: List[float]) -> str:
        """Provide consistent predict() API [cite: 39]"""
        pass
        
    @abstractmethod
    def predict_proba(self, instance: List[float]) -> List[float]:
        """Handles predict_proba for explainability [cite: 41]"""
        pass

# --- 2. Concrete Adapter Implementation ---

class SklearnAdapter(ModelAdapter):
    """Adapts a Scikit-learn model to the ModelAdapter interface."""
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model  # The wrapped sklearn model [cite: 42]
        self._feature_names = feature_names

    def get_feature_names(self) -> List[str]:
        """Returns the feature names used by the model."""
        return self._feature_names

    def predict(self, instance: List[float]) -> str:
        """
        Predicts the class label for a single instance.
        Converts the list input to a format the model expects.
        """
        # Create a DataFrame for consistent input
        input_df = pd.DataFrame([instance], columns=self._feature_names)
        
        prediction_result = self.model.predict(input_df)
        return str(prediction_result[0])

    def predict_proba(self, instance: List[float]) -> List[float]:
        """
        Returns the probability of the prediction.
        """
        input_df = pd.DataFrame([instance], columns=self._feature_names)
        
        # Scikit-learn models usually have predict_proba
        if hasattr(self.model, 'predict_proba'):
            proba_result = self.model.predict_proba(input_df)
            return proba_result[0].tolist()
        
        # Fallback if no proba is available (though rare for classification)
        return []