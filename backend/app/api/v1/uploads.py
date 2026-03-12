import os
import shutil
import tempfile
from typing import List, Dict, Any 
import json
import mlflow
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.app.core.security import decode_access_token
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.services.s3client import upload_fileobj
from backend.app.services.mlflow_client import log_model_registration
from backend.app.db import models
from backend.app.schemas.model import ModelOut, ModelRegister
from backend.app.utils.hashing import sha256_of_fileobj
from backend.app.utils.metrics import REQUEST_COUNT, REQUEST_ERRORS, REQUEST_LATENCY
from contextlib import contextmanager
# from app.api.v1.uploads import get_current_user_id
import io

router = APIRouter()

security_scheme = HTTPBearer(auto_error=True)

def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db)
):
    """Decodes the JWT and validates the user exists."""
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        # User ID exists in token but not DB (shouldn't happen post-register)
        raise HTTPException(status_code=404, detail="User not found")
    
    return user.id

@contextmanager
def _measure(endpoint):
    import time
    start = time.time()
    try:
        yield
    except Exception:
        REQUEST_ERRORS.labels(endpoint).inc()
        raise
    finally:
        REQUEST_COUNT.labels(endpoint, "POST").inc()
        REQUEST_LATENCY.labels(endpoint).observe(time.time() - start)

@router.post("/upload", response_model=ModelOut)
async def upload_model(
    file: UploadFile = File(...), 
    framework: str = "sklearn",
    model_class: str = Query("LogisticRegression", description="Class name of the model (e.g., LogisticRegression)"), # NEW
    framework_path: str = Query("sklearn.linear_model", description="Full module path (e.g., sklearn.linear_model)"),
    features: str = Query(..., description="Comma-separated feature columns"),
    target: str = Query(..., description="Target column name"),
    hyperparams_json: str = Query("{}", description="Hyperparameters as JSON string"),
    model_display_name: str = Query(..., description="User-defined model display name"),
    current_user_id: str = Depends(get_current_user_id), 
    db: Session = Depends(get_db)
    ):
    owner_id_to_use = str(current_user_id)
    endpoint = "/api/v1/models/upload"

    # 1. Prepare Metadata from Query Params
    try:
        hyperparams = json.loads(hyperparams_json)
        feature_list = [f.strip() for f in features.split(',')]
        if not feature_list or not target.strip():
             raise HTTPException(400, "Features and Target must be provided for retraining eligibility.")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format for hyperparameters.")

    with _measure(endpoint):
        if not owner_id_to_use:
            raise HTTPException(400, "owner_id required")
        # Validate owner exists
        owner = db.query(models.User).filter(models.User.id == owner_id_to_use).first()
        if not owner:
            raise HTTPException(404, "owner not found")
        # read file into memory-safe buffer
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        sha256 = sha256_of_fileobj(file_obj)
        # store object under <owner_id>/<sha>.pkl
        object_name = f"{owner_id_to_use}/{sha256}_{model_display_name}"
        try:
            # upload to minio
            upload_fileobj(file_obj, settings.MINIO_BUCKET, object_name)
        except Exception as e:
            raise HTTPException(500, f"Storage error: {e}")
        s3_path = f"s3://{settings.MINIO_BUCKET}/{object_name}"
        # create DB entry
        model_row = models.ModelArtifact(
            name=model_display_name, 
            owner_id=owner.id, 
            framework=framework, 
            artifact_path=s3_path, 
            sha256=sha256,
            meta_info={
                "features": feature_list,
                "target": target.strip(),
                "hyperparams": hyperparams,
                "model_class": model_class,
                "framework_path": framework_path,
            }
            )
        db.add(model_row); db.commit(); db.refresh(model_row)
        # register in MLflow (log artifact: we need to download to local temp then register)
        # write temp local file
        tmpdir = tempfile.mkdtemp()
        tmpfile = os.path.join(tmpdir, file.filename)
        with open(tmpfile, "wb") as f:
            f.write(contents)
        
        upload_params = {
            "owner_id": str(owner.id), 
            "framework": framework, 
            "features": features, 
            "target": target, 
            "hyperparams_json": hyperparams_json
        }

        try:
            log_model_registration(tmpfile, model_row.name, params=upload_params, metrics=None)
        finally:
            shutil.rmtree(tmpdir)
        return model_row

@router.get("/list", response_model=List[ModelOut], tags=["models"])
def list_user_models(
    current_user_id: str = Depends(get_current_user_id), # Requires a valid JWT token
    db: Session = Depends(get_db)
):
    """
    Retrieves a list of all model artifacts uploaded by the authenticated user.
    """
    models_list = db.query(models.ModelArtifact).filter(
        models.ModelArtifact.owner_id == current_user_id
    ).all()
    
    if not models_list:
        # Return an empty list if no models are found, which is standard practice for lists.
        return []
        
    return models_list