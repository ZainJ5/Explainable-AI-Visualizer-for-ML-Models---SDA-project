# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

# --- Request Models (Data sent to the API) ---

class LoadModelRequest(BaseModel):
    """Schema for loading a model."""
    path: str = Field(..., description="Local file path to the pickled ML model.")
    feature_names: List[str] = Field(..., description="List of feature names in order.")
    # background_data is complex (numpy array), we'll simplify its passing or assume it's pre-loaded.
    # For now, we omit it from the request body for simplicity and assume controller handles it.

class FeatureChangeRequest(BaseModel):
    """Schema for changing a single feature via a slider command."""
    feature_index: int = Field(..., description="Index of the feature to change (e.g., 0 for 'Weight').")
    new_value: float = Field(..., description="The new float value for the feature.")

# --- Response/State Models (Data returned from the API) ---

class ModelStatusResponse(BaseModel):
    """Schema for checking the model's current status."""
    is_loaded: bool
    feature_names: List[str]
    current_prediction: Optional[str] = None
    undo_stack_size: int
    redo_stack_size: int