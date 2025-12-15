"""
ExplainerBase - Strategy Pattern Interface
Abstract base class for all explanation algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class ExplainerBase(ABC):
    """
    Strategy Pattern Interface: Defines a common interface for all explanation algorithms.
    
    This abstract class ensures that all explainer implementations provide a consistent
    explain() method, enabling interchangeable explanation strategies.
    
    Responsibilities:
    - Declare explain(instance) method
    - Ensure consistent interface across all explainer types
    """
    
    @abstractmethod
    def explain(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Generate an explanation for a given instance
        
        Args:
            instance: Input instance to explain (numpy array or list)
            
        Returns:
            Dictionary containing explanation data:
            - 'feature_importance': Dict mapping feature names to importance values
            - 'explanation_type': String indicating the type of explanation
            - 'prediction': The model's prediction
            - Additional algorithm-specific data
        """
        pass
