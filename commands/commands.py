# commands/commands.py
from abc import ABC, abstractmethod
from typing import List, Optional

# The Receiver will be the current instance data held by the ModelController
class Command(ABC):
    """
    Purpose: Defines the interface for all commands (Abstract Command).
    """
    @abstractmethod
    def execute(self):
        """Executes the command, changing the state."""
        pass

    @abstractmethod
    def undo(self):
        """Undoes the command, restoring the previous state."""
        pass

# --- Concrete Command Implementation ---

class SliderChangeCommand(Command):
    """
    Purpose: Encapsulates the request to change a single feature value.
    This stores the feature index, the old value, and the new value.
    """
    def __init__(self, controller, feature_index: int, new_value: float):
        # The ModelController acts as the 'Receiver' or context for the change
        self.controller = controller 
        self.feature_index = feature_index
        self.new_value = new_value
        
        # Capture the current value (the 'old state') for undo
        # The controller's current_instance is the array of feature values
        self.old_value = self.controller.get_current_instance()[feature_index]

    def execute(self):
        """Apply the new value and trigger a re-prediction/re-explanation."""
        # The controller updates its internal instance data and notifies observers
        self.controller.set_feature_value(self.feature_index, self.new_value)

    def undo(self):
        """Restore the old value and trigger a re-prediction/re-explanation."""
        # The controller restores the feature to its captured old state
        self.controller.set_feature_value(self.feature_index, self.old_value)