# events/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class Observer(ABC):
    """
    Purpose: Defines update behavior for UI or logging classes (Interface).
    """
    @abstractmethod
    def update(self, data: Dict[str, Any]):
        """
        Receives update on model/explanation changes.
        'data' will typically contain the new prediction and explanation results.
        """
        pass

class ConsoleLogger(Observer):
    """Concrete Observer implementation for testing/logging (stand-in for React components)."""
    def update(self, data: Dict[str, Any]):
        print("\n[Observer Notification Received]")
        print(f"New Prediction: {data.get('prediction', 'N/A')}")
        print(f"Feature Inputs: {data.get('instance', 'N/A')}")
        # This is where the UI component (e.g., a chart) would re-render
        if 'explanation' in data:
            print(f"Explanation Status: {data['explanation']}")