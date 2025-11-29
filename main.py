# main.py
from domain.model_controller import ModelController
from util.create_model import generate_and_save_model
from events.interfaces import ConsoleLogger 
# Import the Command
from commands.commands import SliderChangeCommand
import numpy as np

# A small dummy background dataset
DUMMY_BACKGROUND_DATA = np.array([
    [150.0, 6.5, 0.8],
    [130.0, 5.5, 0.7],
    [170.0, 6.8, 0.88],
])

# Initial Instance to start the interaction
INITIAL_INSTANCE = [155.0, 6.6, 0.82] 


def run_test():
    model_path = 'models/fruit_classifier_tree.pkl'
    FEATURE_NAMES = ['Weight', 'Size', 'Color_Index']
    
    # 1. Get the Singleton instance
    controller = ModelController.get_instance()
    command_manager = controller.command_manager # Access the Invoker

    # --- Observer Pattern Setup ---
    prediction_observer = ConsoleLogger()
    controller.subject.attach(prediction_observer)
    print("ConsoleLogger (Observer) attached to ModelSubject.")
    
    # 2. Load the model and create the Explainer Strategy
    controller.load_model(model_path, FEATURE_NAMES, background_data=DUMMY_BACKGROUND_DATA)
    
    # Initialize the controller's state with a starting point
    controller.current_instance = list(INITIAL_INSTANCE)
    print(f"\n--- Initial State Prediction for {controller.get_current_instance()} ---")
    controller.predict(controller.current_instance) 

    # --- COMMAND PATTERN DEMONSTRATION ---
    print("\n=======================================================")
    print("           Demonstrating Command Pattern (Undo/Redo)   ")
    print("=======================================================")

    # 3. COMMAND 1: Change Weight (Index 0)
    # Move Weight from 155.0 to 195.0 (A decisive Apple feature change)
    feature_index_weight = 0
    new_weight = 195.0
    cmd1 = SliderChangeCommand(controller, feature_index_weight, new_weight)
    print(f"\n[ACTION] Executing Command 1: Change {FEATURE_NAMES[feature_index_weight]} from {cmd1.old_value} to {cmd1.new_value}")
    command_manager.execute_command(cmd1)
    
    # 4. COMMAND 2: Change Size (Index 1)
    # Move Size from 6.6 to 5.0 (A decisive Orange feature change)
    feature_index_size = 1
    new_size = 5.0
    cmd2 = SliderChangeCommand(controller, feature_index_size, new_size)
    print(f"\n[ACTION] Executing Command 2: Change {FEATURE_NAMES[feature_index_size]} from {cmd2.old_value} to {cmd2.new_value}")
    command_manager.execute_command(cmd2)

    # 5. UNDO: Undo Command 2 (Size change)
    print("\n[ACTION] Calling UNDO (Should revert Size to 6.6)")
    command_manager.undo() # This automatically triggers a new prediction/notification

    # 6. UNDO: Undo Command 1 (Weight change)
    print("\n[ACTION] Calling UNDO (Should revert Weight to 155.0, back to initial state)")
    command_manager.undo() # This automatically triggers a new prediction/notification
    
    # 7. REDO: Redo Command 1 (Weight change)
    print("\n[ACTION] Calling REDO (Should re-apply Weight=195.0)")
    command_manager.redo() # This automatically triggers a new prediction/notification
    
    # 8. REDO: Redo Command 2 (Size change)
    print("\n[ACTION] Calling REDO (Should re-apply Size=5.0)")
    command_manager.redo() # This automatically triggers a new prediction/notification


if __name__ == '__main__':
    # Ensure the model is available before running the main test
    # generate_and_save_model() 
    run_test()