"""
SliderChangeCommand - Concrete Command Implementation
Allows undo/redo of feature slider changes in instance input
"""

from typing import List
import numpy as np
from .command import Command


class SliderChangeCommand(Command):
    """
    Concrete Command: Allows undo/redo of feature slider changes.
    
    This command encapsulates a change to a feature value via a slider,
    storing both the old and new values to enable undo/redo operations.
    
    Responsibilities:
    - Change feature value
    - Undo/restore feature value
    - Track state changes
    """
    
    def __init__(self, instance: List[float], feature_index: int, 
                 old_val: float, new_val: float, callback=None):
        """
        Initialize the slider change command
        
        Args:
            instance: The instance array to modify
            feature_index: Index of the feature to change
            old_val: Previous value of the feature
            new_val: New value to set
            callback: Optional callback function to call after execute/undo
        """
        self.instance = instance
        self.feature_index = feature_index
        self.old_val = old_val
        self.new_val = new_val
        self.callback = callback
    
    def execute(self) -> None:
        """
        Execute the slider change by setting the new value
        """
        self.instance[self.feature_index] = self.new_val
        
        if self.callback:
            self.callback()
    
    def undo(self) -> None:
        """
        Undo the slider change by restoring the old value
        """
        self.instance[self.feature_index] = self.old_val
        
        if self.callback:
            self.callback()
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"SliderChangeCommand(feature={self.feature_index}, "
                f"old={self.old_val:.2f}, new={self.new_val:.2f})")
