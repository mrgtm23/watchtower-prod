# backend/app/api/v1/training.py

from fastapi import APIRouter, Depends, status, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.app.api.v1.uploads import get_current_user_id 
from backend.app.services.trainer import run_model_training

router = APIRouter()

class TrainingRequest(BaseModel):
    model_name: str 
    model_class: str 
    framework_path: str
    hyperparams: Dict[str, Any] = {}

@router.post("/train", status_code=status.HTTP_202_ACCEPTED, tags=["training"])
def trigger_training(
    request_data: TrainingRequest,
    background_tasks: BackgroundTasks,
    features: str = Query(..., description="Comma-separated list of feature column names (e.g., 'col1,col2')."), 
    target: str = Query(..., description="The name of the target column in the CSV (e.g., 'label')."), 
    current_user_id: str = Depends(get_current_user_id),
    
):
    user_id_str = str(current_user_id)
    feature_list = [f.strip() for f in features.split(',')]

    print("Function triggered")
    
    # Pass all dynamic arguments to the background task
    background_tasks.add_task(
        run_model_training,
        user_id=user_id_str,
        model_name=request_data.model_name,
        model_class_name=request_data.model_class,
        framework_path=request_data.framework_path,
        hyperparams=request_data.hyperparams,
        feature_columns=feature_list,
        target_column=target
    )

    return {
        "status": "Job Accepted",
        "message": f"Training job for '{request_data.model_name}' ({request_data.model_class}) started. Check MLflow UI.",
        "user_id": user_id_str
    }