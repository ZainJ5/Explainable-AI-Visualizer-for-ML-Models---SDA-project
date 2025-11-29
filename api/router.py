# api/router.py
from fastapi import APIRouter, HTTPException
from domain.model_controller import ModelController
from .schemas import LoadModelRequest, FeatureChangeRequest, ModelStatusResponse
import numpy as np

# 1. Initialize the Router
router = APIRouter()

# 2. Access the Singleton (ModelController)
controller = ModelController.get_instance()

# --- Utility Function (For Demo: Get background data) ---
# In a real app, this would be loaded from a config file or training pipeline.
def get_dummy_background_data():
    return np.array([
        [150.0, 6.5, 0.8],
        [130.0, 5.5, 0.7],
        [170.0, 6.8, 0.88],
    ])

# ==========================================================
# A. Model Initialization and Status
# ==========================================================

@router.post("/load_model", status_code=200)
def load_model_endpoint(request: LoadModelRequest):
    """Loads the ML model and initializes the explainer strategy."""
    try:
        # Load model using the Singleton's method
        controller.load_model(
            path=request.path, 
            feature_names=request.feature_names,
            background_data=get_dummy_background_data() # Using our dummy data
        )
        return {"message": "Model and Explainer successfully loaded."}
    except Exception as e:
        # FastAPI automatically handles the 500 status code
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

@router.get("/status", response_model=ModelStatusResponse)
def get_status_endpoint():
    """Returns the current operational status of the model controller."""
    is_loaded = controller.adapter is not None
    current_prediction = None
    if is_loaded and controller.current_instance:
         # Make a quick prediction based on the current state (last prediction's result)
         current_prediction = controller.predict(controller.current_instance) 
         
    return ModelStatusResponse(
        is_loaded=is_loaded,
        feature_names=controller.get_feature_names(),
        current_prediction=current_prediction,
        undo_stack_size=len(controller.command_manager.undo_stack),
        redo_stack_size=len(controller.command_manager.redo_stack)
    )

# ==========================================================
# B. Command Layer (Execute, Undo, Redo)
# ==========================================================

@router.post("/execute_command", status_code=200)
def execute_command_endpoint(request: FeatureChangeRequest):
    """
    Executes a new SliderChangeCommand, triggers a re-prediction, and notifies Observers.
    """
    from commands.commands import SliderChangeCommand
    
    # 1. Create the Concrete Command object
    command = SliderChangeCommand(
        controller=controller, 
        feature_index=request.feature_index, 
        new_value=request.new_value
    )
    
    # 2. Hand the Command to the Invoker (CommandManager)
    controller.command_manager.execute_command(command)
    
    # 3. The prediction/explanation/notification is handled internally by the Command's execute()
    return {"message": "Command executed successfully. Check Observer logs for update."}

@router.post("/undo", status_code=200)
def undo_command_endpoint():
    """Calls the CommandManager to undo the last action."""
    controller.command_manager.undo()
    return {"message": "Undo executed. Check Observer logs for state change."}

@router.post("/redo", status_code=200)
def redo_command_endpoint():
    """Calls the CommandManager to redo the last undone action."""
    controller.command_manager.redo()
    return {"message": "Redo executed. Check Observer logs for state change."}