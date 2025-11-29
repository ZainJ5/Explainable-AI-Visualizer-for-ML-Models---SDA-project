# domain/model_controller.py
import pickle
from typing import List, Optional, Any, Dict
from .explainer_factory import ExplainerFactory
from .explainer_strategies import ExplainerBase
from .model_adapter import ModelAdapter, SklearnAdapter
from events.model_subject import ModelSubject 
# Import the CommandManager
from commands.command_manager import CommandManager

class ModelController:
    """
    Purpose: Ensures only one model is loaded and handled throughout the app (Singleton). [cite: 17]
    """
    _instance: Optional['ModelController'] = None
    explainer: Optional[ExplainerBase] = None
    subject: ModelSubject
    command_manager: CommandManager
    current_instance: List[float] = []
    
    # Attributes [cite: 23-26]
    model: any = None
    adapter: Optional[ModelAdapter] = None
    feature_names: List[str] = []

    def __new__(cls, *args, **kwargs):
        """Standard Singleton implementation to control instantiation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.subject = ModelSubject()
            cls._instance.command_manager = CommandManager()
            # Initialize attributes here (or in load_model)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'ModelController':
        """Provides a single, global point of access for the model. [cite: 216]"""
        return cls._instance or cls()

    def load_model(self, path: str, feature_names: List[str],background_data: Optional[Any] = None):
        """Load ML model and wrap it with an Adapter. [cite: 18, 28]"""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Use the Adapter to wrap the raw model
            self.adapter = SklearnAdapter(self.model, feature_names)
            self.feature_names = feature_names
            print(f"Model successfully loaded from {path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.adapter = None
        if self.adapter:
            # Initialize the current instance with a neutral value (e.g., all zeros or an example)
            # For simplicity, we'll initialize it with zeros if not set externally.
            if not self.current_instance:
                 self.current_instance = [0.0] * len(feature_names) 
            
            # Clear command history for a new model
            self.command_manager.clear()
        if self.adapter:
            try:
                # 1. Create the Explainer using the Factory Method
                self.explainer = ExplainerFactory.create(self.adapter, background_data)
                print(f"Explainer strategy set to: {self.explainer.__class__.__name__}")
            except Exception as e:
                print(f"Warning: Could not create explainer: {e}")
                self.explainer = None
    def get_current_instance(self) -> List[float]:
        return self.current_instance
        
    # New method: The command calls this to modify the state
    def set_feature_value(self, index: int, value: float):
        """
        Updates a feature value in the current instance and triggers
        a prediction/explanation update through the existing pipeline.
        """
        if 0 <= index < len(self.current_instance):
            self.current_instance[index] = value
            # Crucially, calling predict() now uses the updated current_instance
            # and automatically notifies the observers.
            self.predict(self.current_instance) 
            # Note: We return the prediction from predict(), but here we just rely on the side effect
            # of updating the internal state and notifying the subject.

    def predict(self, instance: List[float]) -> str:
        """Provide predictions using the Adapter. [cite: 20, 30]"""
        if not self.adapter:
            return "Model not loaded"
        self.current_instance = instance
        prediction_result = self.adapter.predict(instance)
        explanation_data: Dict[str, Any] = {}
        if self.explainer:
            # 2. Execute the Strategy's explain() method (Polymorphism)
            explanation_data = self.explainer.explain(instance)
        update_data: Dict[str, Any] = {
            "instance": instance,
            "prediction": prediction_result,
            "explanation_type": explanation_data.get('type', 'None'), 
            "explanation": explanation_data # Send the full explanation object
        }
        
        # Notify the UI layer (Observers) of the change
        self.subject.notify(update_data)
        
        return prediction_result
    

    def get_feature_names(self) -> List[str]:
        """Expose model metadata (feature names). [cite: 19, 26]"""
        return self.feature_names