import os
import tempfile
import joblib
import numpy as np
import time
import io
import json
import shutil
import uuid
from fastapi import APIRouter, HTTPException, Depends
from backend.app.services.s3client import download_to_file, upload_fileobj
from backend.app.core.config import settings
from backend.app.utils.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUEST_ERRORS, MODEL_INFERENCE_LATENCY
from datetime import datetime
from backend.app.api.v1.uploads import get_current_user_id
from backend.app.db.session import get_db # ADD
from backend.app.db import models # ADD
from sqlalchemy.orm import Session # ADD

router = APIRouter()

@router.post("/models/predict/{owner_id}/{model_sha_and_name}")
def predict(
    owner_id: str, 
    model_sha_and_name: str, 
    payload: dict, 
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
    ):
    endpoint = "/api/v1/models/predict"
    model_sha = model_sha_and_name.split('_')[0]
    start = time.time()
    
    # --- Start Core Inference Timer ---
    start_inference = time.time()
    tmpdir = None
    try:
        # 1. Look up model artifact in the database
        model_entry = db.query(models.ModelArtifact).filter(
            models.ModelArtifact.sha256 == model_sha,
            models.ModelArtifact.owner_id == owner_id 
        ).first()

        if not model_entry:
            raise HTTPException(404, f"Model artifact metadata not found in DB for SHA: {model_sha}")
        
        if model_entry.framework == "Scikit-learn": # Check if it's an MLflow-registered model
            # For MLflow trained models, the artifact path is the S3 directory path.
            # We must use MLflow to download the model since it handles the complex packaging.
            import mlflow.pyfunc
            
            # The artifact_path stored in DB is the URI (e.g., s3://bucket/mlruns/0/run_id/artifacts/model_artifact)
            s3_uri = model_entry.artifact_path
            
            # MLflow requires downloading to a local temporary directory first
            tmpdir = tempfile.mkdtemp()
            
            # Use MLflow's utility to load the model from the S3 URI into the temporary directory
            # mlflow.pyfunc.load_model returns the model wrapper object, but we need the artifact files locally first.
            
            # We must use the MLflow client to download the artifact directory.
            client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_URI)
            
            # Extract the MLflow Run ID from the SHA for downloading (e.g., MLFLOW-5dd7a493... -> 5dd7a493...)
            print(f"model_sha ::::::{model_sha}")
            run_id = model_sha.split('-')[-1]
            print(f"run id ::::::{run_id}")
            # Download the entire model artifact folder to the local temp directory
            client.download_artifacts(
                run_id=run_id, 
                path="model_artifact", 
                dst_path=tmpdir
            )
            
            # Load the model from the local directory structure using MLflow's pyfunc
            # The model is now loaded as a generic MLflow Python Function
            model_path_in_temp = os.path.join(tmpdir, "model_artifact")
            model = mlflow.pyfunc.load_model(model_path_in_temp)
        else:
            # model object path in S3: s3://<bucket>/<owner_id>/<sha>_<filename>
            object_name = f"{owner_id}/{model_sha_and_name}"
            tmpdir = tempfile.mkdtemp()
            local_file = os.path.join(tmpdir, model_sha_and_name)
            try:
                download_to_file(settings.MINIO_BUCKET, object_name, local_file)
            except Exception as e:
                raise HTTPException(404, f"Model not found: {e}")
            # load model
            model = joblib.load(local_file)
        # expect payload {"input": [ ... ]}
        if "input" not in payload:
            raise HTTPException(400, "payload must contain 'input' key")
        arr = np.array(payload["input"]).reshape(1, -1)
        pred = model.predict(arr).tolist()[0]
        # log lightweight prediction event to file
        # 4. Log prediction event to S3/MinIO
        log_data = json.dumps({
            "ts": datetime.utcnow().isoformat(), 
            "owner": owner_id, 
            "model_sha": model_sha, 
            "model": model_sha_and_name, 
            "input": payload["input"], 
            "pred": pred
        }) + "\n"

        # Define the S3 object path (Isolated log)
        log_object_name = f"logs/{model_sha}/inference.log" # Or logs/inference_{model_sha}.log

        log_entry_object = f"logs/{model_sha}/entry_{uuid.uuid4()}.json"

        try:
            log_file_obj = io.BytesIO(log_data.encode('utf-8'))
            # Use upload_fileobj (from s3client)
            upload_fileobj(log_file_obj, settings.MINIO_BUCKET, log_entry_object)
            
        except Exception as e:
            # Log the upload error, but don't crash the prediction response
            print(f"Failed to upload log entry to MinIO: {e}")

        # os.makedirs("logs", exist_ok=True)
        # log_file_path = f"logs/inference_{model_sha}.log"
        # with open(log_file_path, "a") as f:
        #     f.write(json.dumps({"ts": datetime.utcnow().isoformat(), "owner": owner_id, "model_sha": model_sha, "model": model_sha_and_name, "input": payload["input"], "pred": pred}) + "\n")

        # --- Stop Core Inference Timer ---
        inference_time = time.time() - start_inference
        MODEL_INFERENCE_LATENCY.labels(model_sha=model_sha.split('_')[0]).observe(inference_time)

        return {"prediction": pred}
    except Exception as e:
        REQUEST_ERRORS.labels(endpoint).inc()
        raise HTTPException(500, f"Inference processing error: {e}")
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        REQUEST_COUNT.labels(endpoint, "POST").inc()
        REQUEST_LATENCY.labels(endpoint).observe(time.time() - start)