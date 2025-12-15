"""
ModelSubject - Observable Pattern Implementation
Notifies observers when model predictions or explanations change
"""

from typing import List, Dict, Any
from .observer import Observer


class ModelSubject:
    """
    Observable Pattern: Notify UI observers when predictions or explanations change.
    
    This class maintains a list of observers and notifies them whenever there are
    changes to model state, predictions, or explanations.
    
    Responsibilities:
    - Maintain observer list
    - Notify on change
    - Manage observer registration/deregistration
    """
    
    def __init__(self):
        """Initialize the subject with an empty observer list"""
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the subject
        
        Args:
            observer: Observer instance to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the subject
        
        Args:
            observer: Observer instance to detach
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, data: Dict[str, Any]) -> None:
        """
        Notify all observers of a change
        
        Args:
            data: Dictionary containing notification data:
                  - 'type': Type of notification ('prediction', 'explanation', 'model_loaded', 'error')
                  - Additional context-specific data
        """
        for observer in self._observers:
            try:
                observer.update(data)
            except Exception as e:
                print(f"Error notifying observer: {str(e)}")
    
    def notify_model_loaded(self, feature_names: List[str]) -> None:
        """
        Convenience method to notify observers that a model has been loaded
        
        Args:
            feature_names: List of feature names from the loaded model
        """
        self.notify({
            'type': 'model_loaded',
            'feature_names': feature_names
        })
    
    def notify_prediction(self, prediction: Any, probabilities: Any = None, instance: Any = None) -> None:
        """
        Convenience method to notify observers of a new prediction
        
        Args:
            prediction: The model's prediction
            probabilities: Prediction probabilities (if available)
            instance: The input instance
        """
        self.notify({
            'type': 'prediction',
            'prediction': prediction,
            'probabilities': probabilities,
            'instance': instance
        })
    
    def notify_explanation(self, explanation: Dict[str, Any]) -> None:
        """
        Convenience method to notify observers of a new explanation
        
        Args:
            explanation: Dictionary containing explanation data from an explainer
        """
        self.notify({
            'type': 'explanation',
            'explanation': explanation
        })
    
    def notify_error(self, error_message: str) -> None:
        """
        Convenience method to notify observers of an error
        
        Args:
            error_message: Description of the error
        """
        self.notify({
            'type': 'error',
            'message': error_message
        })
