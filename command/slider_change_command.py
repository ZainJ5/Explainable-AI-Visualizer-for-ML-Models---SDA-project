"""
Command Pattern Implementation
Abstract base class for all undoable commands
"""

from abc import ABC, abstractmethod


class Command(ABC):
    """
    Command Pattern Interface: Define interface for undoable operations.
    
    This abstract class ensures that all command implementations provide
    both execute() and undo() methods, enabling full undo/redo functionality.
    
    Responsibilities:
    - Define executable operation
    - Define undoable operation
    - Ensure consistent command interface
    """
    
    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command
        
        This method performs the primary action of the command.
        """
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """
        Undo the command
        
        This method reverses the action performed by execute(),
        restoring the previous state.
        """
        pass
