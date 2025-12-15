"""
ModelController - Singleton Pattern Implementation
Manages the ML model lifecycle and ensures only one model instance exists
"""

from typing import List, Union
import numpy as np
from .model_adapter import ModelAdapter
from .model_loader_strategy import ModelLoaderContext
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass  


class ModelController:
    """
    Singleton Pattern: Ensures only one ML model is loaded and handled throughout the app.
    
    Responsibilities:
    - Load ML model
    - Expose model metadata
    - Provide predictions
    - Maintain a single global model instance
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton implementation - ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(ModelController, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the controller (only once due to Singleton)"""
        if self._initialized:
            return
            
        self.model = None
        self.adapter: ModelAdapter = None
        self.feature_names: List[str] = []
        self._initialized = True
    
    def load_model(self, path: str) -> List[str]:
        """
        Load an ML model from a file using Strategy Pattern for compatibility
        
        Args:
            path: Path to the model file (.pkl, .json, etc.)
            
        Returns:
            List of feature names
        """
        try:
            if path.endswith('.pkl') or path.endswith('.pickle'):
                loader_context = ModelLoaderContext()
                self.model = loader_context.load_model(path)
            
            elif path.endswith('.json'):
                with open(path, 'r') as f:
                    model_data = json.load(f)
                    self.model = model_data  
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            self.adapter = ModelAdapter(self.model)
            self.feature_names = self.adapter.get_feature_names()
            
            return self.feature_names
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, instance: Union[np.ndarray, List]) -> tuple:
        """
        Make a prediction using the loaded model
        
        Args:
            instance: Input instance (list or numpy array)
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        if self.adapter is None:
            raise Exception("No model loaded. Please load a model first.")
        
        return self.adapter.predict(instance)
    
    def get_model(self):
        """
        Get the current model instance
        
        Returns:
            The loaded model object
        """
        return self.model
    
    def get_adapter(self) -> ModelAdapter:
        """
        Get the model adapter
        
        Returns:
            ModelAdapter instance
        """
        return self.adapter
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
