# domain/explainer_strategies.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# We use the abstract base class as the Strategy Interface
class ExplainerBase(ABC):
    """
    Purpose: Defines a common interface for all explanation algorithms (Strategy Interface).
    Achieves Liskov Substitution Principle (LSP).
    """
    def __init__(self, model_adapter, feature_names: List[str]):
        self.adapter = model_adapter  # The model wrapper
        self.feature_names = feature_names

    @abstractmethod
    def explain(self, instance: List[float]) -> Dict[str, Any]:
        """
        Calculates and returns the explanation for a single instance.
        The output format must be consistent across all strategies.
        """
        pass

# --- Concrete Strategy Implementations ---

class DecisionPathExplainer(ExplainerBase):
    """
    Strategy for tree-based models (like DecisionTreeClassifier).
    Explains the prediction by showing the path taken down the tree.
    """
    def explain(self, instance: List[float]) -> Dict[str, Any]:
        
        # Get the underlying sklearn model from the adapter
        tree_model = self.adapter.model 
        
        # Convert instance list to numpy array for sklearn
        instance_np = np.array(instance).reshape(1, -1)
        
        # Find the leaf node ID for the instance
        node_indicator = tree_model.decision_path(instance_np)
        leaf_id = tree_model.apply(instance_np)[0]
        
        # Extract rules leading to the leaf
        feature = tree_model.tree_.feature
        threshold = tree_model.tree_.threshold
        
        rules = []
        for node_id in node_indicator.indices:
            if leaf_id == node_id:
                break
            
            # Extract feature name and comparison
            if instance_np[0, feature[node_id]] <= threshold[node_id]:
                comparison = f"{self.feature_names[feature[node_id]]} <= {threshold[node_id]:.2f}"
            else:
                comparison = f"{self.feature_names[feature[node_id]]} > {threshold[node_id]:.2f}"
                
            rules.append(comparison)

        return {
            "type": "Decision Path",
            "description": "Rules leading to the prediction:",
            "rules": rules
        }

class SHAPExplainer(ExplainerBase):
    """
    Strategy for model-agnostic explanation using SHAP.
    Requires background data (training data) for proper operation.
    """
    def __init__(self, model_adapter, feature_names: List[str], background_data: np.ndarray):
        super().__init__(model_adapter, feature_names)
        import shap 
        # Using a model-agnostic method like KernelExplainer or explainer for tree
        # Since our model is a tree, we use TreeExplainer for efficiency.
        self.explainer = shap.TreeExplainer(self.adapter.model)

    def explain(self, instance: List[float]) -> Dict[str, Any]:
        instance_np = np.array(instance).reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance_np)
        
        # For classification, shap_values is a list of arrays (one for each class)
        # We'll use the explanation for the predicted class (assuming binary classification)
        predicted_class_index = np.argmax(self.adapter.model.predict_proba(instance_np))
        feature_contributions = shap_values[predicted_class_index][0]

        # Combine feature names and SHAP values
        explanation_data = {
            self.feature_names[i]: float(feature_contributions[i])
            for i in range(len(self.feature_names))
        }

        # Sort for display (optional, but helpful for UI)
        sorted_contributions = sorted(explanation_data.items(), key=lambda item: abs(item[1]), reverse=True)

        return {
            "type": "SHAP Values",
            "description": "Feature contributions (SHAP values):",
            "contributions": sorted_contributions
        }