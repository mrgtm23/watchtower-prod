# backend/app/api/v1/monitoring.py (UPDATED)

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session 
from backend.app.db.session import get_db 
from backend.app.db import models 
from backend.app.api.v1.uploads import get_current_user_id 
from scripts.drift_monitor import generate_drift_report 
from backend.app.utils.hashing import sha256_of_fileobj 
from backend.app.services.s3client import upload_fileobj 
from backend.app.core.config import settings 
from backend.app.services.trainer import run_model_training # NEW IMPORT for retraining
from pydantic import BaseModel, Field
from typing import List, Union
from datetime import datetime
import os
import shutil
import pandas as pd
import io
import json
import uuid

router = APIRouter()

# --- Schemas for Listing (Add this class) ---
class DatasetOut(BaseModel):
    # This must match the DB model columns
    id: uuid.UUID
    name: str
    artifact_path: str
    sha256: str
    created_at: datetime
    
    class Config: # Needed for ORM compatibility
        orm_mode = True

# REFERENCE_BASE_DIR = "scripts/reference_data"

@router.post("/reference", status_code=status.HTTP_201_CREATED)
async def upload_reference_dataset(
    file: UploadFile = File(..., description="The CSV file containing the model's training/reference dataset for drift monitoring."),
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    
    if file.content_type != 'text/csv':
        raise HTTPException(400, f"Invalid file type. Only CSV files are accepted. {file.content_type}")
    
    """ Uploads a reference dataset (CSV) to MinIO and registers its path in the DB. """
    user_id_str = str(current_user_id)
    

    # """
    # Uploads a reference dataset (CSV) for a user's model monitoring.
    # File is stored under: scripts/reference_data/{user_id}/reference.csv
    # """
    # if file.content_type != 'text/csv':
    #     raise HTTPException(400, f"Invalid file type. Only CSV files are accepted. {file.content_type}")

    # user_id_str = str(current_user_id)
    # user_dir = os.path.join(REFERENCE_BASE_DIR, user_id_str)
    
    # os.makedirs(user_dir, exist_ok=True)
    # destination_path = os.path.join(user_dir, "reference.csv")

    contents = await file.read()
    file_obj = io.BytesIO(contents)

    try:
        # Simple validation: ensure pandas can read the CSV
        file_obj.seek(0)
        pd.read_csv(file_obj)

        # 2. Hashing and MinIO Storage
        file_obj.seek(0)
        sha256 = sha256_of_fileobj(file_obj)
        dataset_uuid = str(uuid.uuid4())
        
        # Object name: <owner_id>/datasets/<dataset_uuid>_<filename>
        object_name = f"{user_id_str}/datasets/{dataset_uuid}_{file.filename}"
        file_obj.seek(0)
        upload_fileobj(file_obj, settings.MINIO_BUCKET, object_name)
        s3_path = f"s3://{settings.MINIO_BUCKET}/{object_name}"

        # 3. Database Registration
        dataset_row = models.ReferenceDataset(
            id=dataset_uuid,
            name=file.filename, 
            owner_id=user_id_str, 
            artifact_path=s3_path, 
            sha256=sha256
        )
        db.add(dataset_row); db.commit(); db.refresh(dataset_row)

        return {"message": "Reference dataset uploaded and registered successfully", "dataset_id": dataset_uuid}
        
        # with open(destination_path, "wb") as f:
        #     f.write(contents)

        # return {"message": "Reference dataset uploaded successfully", "path": destination_path}
    
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "Uploaded CSV file is empty.")
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {e}")
    
@router.get("/reference/list", response_model=List[DatasetOut], tags=["monitoring"])
def list_reference_datasets(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """ Retrieves a list of all reference datasets uploaded by the authenticated user. """
    datasets_list = db.query(models.ReferenceDataset).filter(
        models.ReferenceDataset.owner_id == current_user_id
    ).all()
    
    return datasets_list

# --- NEW ENDPOINT: Trigger Drift Check ---
@router.get("/drift-check", tags=["monitoring"], response_class=HTMLResponse)
def trigger_drift_check(
    current_user_id: str = Depends(get_current_user_id),
    features: str = Query(..., description="Comma-separated list of feature column names (e.g., 'col1,col2,col3')."),
    dataset_id: str = Query(..., description="The UUID of the registered reference dataset."),
    model_sha: str = Query(..., description="The SHA of the model whose logs should be checked."),
    db: Session = Depends(get_db)
):
    """ Triggers the Evidently Data Drift check using the specified dataset ID and model SHA.
    Convert comma-separated string to a list of strings """
    feature_list = [f.strip() for f in features.split(',')]

    # 1. Look up the dataset S3 path
    dataset = db.query(models.ReferenceDataset).filter(
        models.ReferenceDataset.id == dataset_id,
        models.ReferenceDataset.owner_id == current_user_id
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Reference dataset not found for this user.")
    
    # report_result = generate_drift_report(
    #     user_id=str(current_user_id),
    #     feature_columns=feature_list
    # )

    # 2. Generate report using MinIO path and isolated log
    report_result = generate_drift_report(
        user_id=str(current_user_id),
        model_sha=model_sha, 
        reference_s3_path=dataset.artifact_path, 
        feature_columns=feature_list
    )
    
    if isinstance(report_result, dict) and 'error' in report_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=report_result['error']
        )
        
    return HTMLResponse(content=report_result)

# --- NEW ENDPOINT: Feature 4 - Retraining Trigger ---
@router.post("/retrain/{model_id}", status_code=status.HTTP_202_ACCEPTED, tags=["training"])
def trigger_retrain(
    model_id: str,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """ Triggers a model retraining job using the metadata stored for the given model_id. """
    
    model_row = db.query(models.ModelArtifact).filter(
        models.ModelArtifact.id == model_id,
        models.ModelArtifact.owner_id == current_user_id
    ).first()
    
    if not model_row:
        raise HTTPException(status_code=404, detail="Model not found or access denied.")

    # 2. Extract necessary training parameters from meta_info
    meta = model_row.meta_info
    
    # Check if a reference dataset ID is stored or assume the latest one?
    # For now, we assume the user must select the data during training, 
    # but the logic pulls all other metadata (features, target, hyperparams).
    
    required_params = {
        "model_name": f"{model_row.name}_RETRAINED_{datetime.now().strftime('%Y%m%d%H%M')}",
        "model_class_name": meta.get("model_class"),
        "framework_path": meta.get("framework_path"),
        "hyperparams": meta.get("hyperparams", {}),
        "feature_columns": meta.get("features"),
        "target_column": meta.get("target")
    }
    
    if not all(required_params.values()):
        raise HTTPException(status_code=400, detail="Missing required training metadata.")

    # 3. Trigger the background training task
    # Note: run_model_training expects 'user_id' as the first arg
    background_tasks.add_task(
        run_model_training,
        user_id=str(current_user_id),
        **required_params
    )

    return {
        "status": "Job Accepted",
        "message": f"Retraining job for '{model_row.name}' started using stored parameters.",
        "original_model_id": model_id
    }