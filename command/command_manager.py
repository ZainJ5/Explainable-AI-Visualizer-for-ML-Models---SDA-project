"""
CommandManager - Invoker Pattern Implementation
Handles execution, undo, and redo of commands
"""

from typing import List
from .command import Command


class CommandManager:
    """
    Invoker Pattern: Handles execution, undo, redo stacks.
    
    This class manages the lifecycle of commands, maintaining separate stacks
    for undo and redo operations to enable full command history navigation.
    
    Responsibilities:
    - Store Command objects
    - Execute commands
    - Manage undo/redo stacks
    - Provide command history
    """
    
    def __init__(self):
        """Initialize the command manager with empty stacks"""
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []
    
    def execute(self, cmd: Command) -> None:
        """
        Execute a command and add it to the undo stack
        
        Args:
            cmd: Command to execute
        """
        try:
            cmd.execute()
            self.undo_stack.append(cmd)
            # Clear redo stack when a new command is executed
            self.redo_stack.clear()
            print(f"Command executed: {cmd}")
        except Exception as e:
            print(f"Error executing command: {str(e)}")
    
    def undo(self) -> bool:
        """
        Undo the most recent command
        
        Returns:
            True if undo was successful, False if undo stack is empty
        """
        if not self.undo_stack:
            print("Nothing to undo")
            return False
        
        try:
            cmd = self.undo_stack.pop()
            cmd.undo()
            self.redo_stack.append(cmd)
            print(f"Command undone: {cmd}")
            return True
        except Exception as e:
            print(f"Error undoing command: {str(e)}")
            return False
    
    def redo(self) -> bool:
        """
        Redo the most recently undone command
        
        Returns:
            True if redo was successful, False if redo stack is empty
        """
        if not self.redo_stack:
            print("Nothing to redo")
            return False
        
        try:
            cmd = self.redo_stack.pop()
            cmd.execute()
            self.undo_stack.append(cmd)
            print(f"Command redone: {cmd}")
            return True
        except Exception as e:
            print(f"Error redoing command: {str(e)}")
            return False
    
    def can_undo(self) -> bool:
        """
        Check if undo is available
        
        Returns:
            True if there are commands to undo
        """
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """
        Check if redo is available
        
        Returns:
            True if there are commands to redo
        """
        return len(self.redo_stack) > 0
    
    def clear(self) -> None:
        """
        Clear both undo and redo stacks
        """
        self.undo_stack.clear()
        self.redo_stack.clear()
        print("Command history cleared")
    
    def get_undo_count(self) -> int:
        """Get the number of undoable commands"""
        return len(self.undo_stack)
    
    def get_redo_count(self) -> int:
        """Get the number of redoable commands"""
        return len(self.redo_stack)
