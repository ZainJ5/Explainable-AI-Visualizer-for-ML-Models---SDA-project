"""
Observer Pattern Implementation
Defines the Observer interface for event-driven updates
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Observer(ABC):
    """
    Observer Interface: Defines update behavior for UI or logging classes.
    
    This interface ensures that all observer classes implement the update method,
    enabling them to receive notifications from Observable subjects.
    
    Responsibilities:
    - Define update(data) interface
    - Ensure consistent notification handling
    """
    
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """
        Receive update from the subject
        
        Args:
            data: Dictionary containing update information:
                  - 'type': Type of update ('prediction', 'explanation', 'model_loaded')
                  - Additional context-specific data
        """
        pass
