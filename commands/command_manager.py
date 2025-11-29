# commands/command_manager.py
from typing import List, Optional
from .commands import Command

class CommandManager:
    """
    Purpose: Manages command history and enables undo/redo functionality (Invoker).
    """
    def __init__(self):
        self.undo_stack: List[Command] = []
        self.redo_stack: List[Command] = []

    def execute_command(self, command: Command):
        """
        Executes a new command, pushes it to the undo stack, and clears the redo stack.
        """
        command.execute()
        self.undo_stack.append(command)
        # Any new action clears the possibility of 'redoing' a previous action
        self.redo_stack.clear()
        print(f"\n[Command] Executed: {command.__class__.__name__}. Stacks: Undo={len(self.undo_stack)}, Redo={len(self.redo_stack)}")

    def undo(self):
        """
        Pops the last command from the undo stack, calls its undo() method,
        and pushes it onto the redo stack.
        """
        if not self.undo_stack:
            print("[Command] Undo stack is empty.")
            return

        command = self.undo_stack.pop()
        command.undo()
        self.redo_stack.append(command)
        print(f"\n[Command] Undid: {command.__class__.__name__}. Stacks: Undo={len(self.undo_stack)}, Redo={len(self.redo_stack)}")
        
    def redo(self):
        """
        Pops the last command from the redo stack, calls its execute() method,
        and pushes it back onto the undo stack.
        """
        if not self.redo_stack:
            print("[Command] Redo stack is empty.")
            return

        command = self.redo_stack.pop()
        command.execute()
        self.undo_stack.append(command)
        print(f"\n[Command] Redid: {command.__class__.__name__}. Stacks: Undo={len(self.undo_stack)}, Redo={len(self.redo_stack)}")

    def clear(self):
        """Clear all stacks."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        print("[Command] Command history cleared.")