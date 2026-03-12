# backend/app/services/trainer.py (FINAL, MINIO-BASED VERSION)

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import os
import uuid
import shutil
import importlib 
import time
from backend.app.db.session import SessionLocal 
from backend.app.db import models 
import mlflow.tracking
from backend.app.services.s3client import download_fileobj # Fetches data from MinIO
from backend.app.core.config import settings # Used for MINIO_BUCKET name
import io # Used to download data into memory
from typing import List, Dict, Any

DEBUG_LOG_PATH = "/tmp/watchtower_training_debug.log"
# Configure logger at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Relies on Docker/ENV) ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("WatchTower_Training_Runs")

def run_model_training(
    user_id: str, 
    model_name: str, 
    model_class_name: str, 
    framework_path: str, 
    hyperparams: Dict[str, Any], 
    feature_columns: List[str],
    target_column: str
) -> dict:
    """Loads user's data from MinIO/DB, runs training, and registers the final artifact."""

    print("--- Starting model training process (MinIO/DB flow)... ---")
    
    # 1. Debug Logging Helper
    # def debug_log(message):
    #     try:
    #         with open(DEBUG_LOG_PATH, 'a') as f:
    #             f.write(f"[DEBUG] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    #     except Exception:
    #         pass # Ignore file write failures in debug logger

    # debug_log("START: Training job initiated.")

    try:
        # --- A. Dynamic Model Import and Initialization ---
        try:
            module = importlib.import_module(framework_path)
            ModelClass = getattr(module, model_class_name)
            model = ModelClass(**hyperparams)
            print("STEP 1: Model imported successfully.")
        except (ImportError, AttributeError) as e:
            return {"status": "failed", "message": f"Model Class Error: Could not import {model_class_name} from {framework_path}. Details: {e}"}
        except TypeError as e:
            return {"status": "failed", "message": f"Hyperparameter Error: Invalid parameters for {model_class_name}. Details: {e}"}
        
        # --- B. Load Data from MinIO (Via DB Lookup) ---
        try:
            # 1. Find the latest reference dataset uploaded by the user
            db = SessionLocal()
            latest_dataset = db.query(models.ReferenceDataset).filter(
                models.ReferenceDataset.owner_id == user_id
            ).order_by(models.ReferenceDataset.created_at.desc()).first()
            db.close()
            print(f"-->latest_dataset<--{latest_dataset}")

            if not latest_dataset:
                return {"status": "failed", "message": "Reference data not found in DB. Upload a CSV first."}

            s3_path = latest_dataset.artifact_path
            print(f"-->s3_path<--{s3_path}")
            
            # 2. Download data from MinIO (S3) into memory
            bucket_name = settings.MINIO_BUCKET
            object_name = s3_path.split(f"s3://{bucket_name}/")[-1]

            print(f"-->new executionnnnn<--{bucket_name} and {object_name}")
            try:
                file_obj = io.BytesIO()
                download_fileobj(bucket_name, object_name, file_obj)
            except Exception as e:
                print(f"error:::{e}")
                return {"status": "failed", "message": f"MinIO/Data loading error: {e}. Check network and bucket setup."}
            
            print(f"-->file_obj<--{file_obj}")
            file_obj.seek(0)
            
            # 3. Load into Pandas and split
            print("Step 3 execution")
            data = pd.read_csv(file_obj)
            print(f"-->data<--{data}")
            print(f"STEP 2: Data loaded from MinIO (Path: {s3_path}).")
            print(f"x::{data[feature_columns]}")
            print(f"x::{data[target_column]}")
            X = data[feature_columns]
            y = data[target_column]
            print(f"x::{X}")
            print(f"y::{y}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print("STEP 3: Data slicing successful. Entering MLflow block...")
            
        except KeyError as e:
            return {"status": "failed", "message": f"Data error: Missing column {e} in MinIO dataset. Check feature/target names."}
        except Exception as e:
            return {"status": "failed", "message": f"MinIO/Data loading error: {e}. Check network and bucket setup."}

        # --- C. MLflow Tracking, Training, and Logging ---
        with mlflow.start_run(run_name=f"Training_{model_class_name}_{user_id[:4]}") as run:
            print("STEP 4: MLflow run successfully started.")
            framework_logged = "Unknown"
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"MLflow run complete. Accuracy: {accuracy}")

            # Log Params
            mlflow.log_params(hyperparams)
            mlflow.log_param("model_class", model_class_name)
            mlflow.log_metric("accuracy", accuracy)

            # 4. Save and Log Artifact (MLflow)
            try:
                # Determine framework and log model artifact
                if "sklearn" in framework_path:
                    mlflow.sklearn.log_model(sk_model=model, artifact_path="model_artifact", registered_model_name=f"{model_class_name}_{model_name}")
                    framework_logged = "Scikit-learn"
                elif "xgboost" in framework_path:
                    mlflow.xgboost.log_model(xgb_model=model, artifact_path="model_artifact", registered_model_name=f"{model_class_name}_{model_name}")
                    framework_logged = "XGBoost"
                else:
                    logger.warning(f"Unsupported framework '{framework_path}'. Artifact registered generically.")
                    framework_logged = framework_path
                    
                logger.info(f"Model artifact successfully logged to MLflow via {framework_logged}.")

            except Exception as e:
                # Do NOT return fail here, registration might still proceed if MLflow fails.
                logger.error(f"Failed to log model artifact to MinIO/MLflow: {e}", exc_info=True)


            # --- D. Automated Model Registration into FastAPI DB ---
            try:
                run_id = run.info.run_id
                
                client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                artifact_uri = client.get_run(run_id).info.artifact_uri
                
                s3_artifact_path = f"{artifact_uri}/model_artifact"
                model_sha = f"MLFLOW-{run_id}" 

                db = SessionLocal()
                model_row = models.ModelArtifact(
                    name=model_name,
                    owner_id=user_id, 
                    framework=framework_logged, 
                    artifact_path=s3_artifact_path,
                    sha256=model_sha,
                    meta_info={
                        "mlflow_run_id": run_id,
                        "accuracy": accuracy,
                        "features": feature_columns,
                        "target": target_column,
                        "hyperparams": hyperparams,
                        "model_class": model_class_name,
                        "framework_path": framework_path
                    }
                )
                db.add(model_row); db.commit(); db.refresh(model_row)
                db.close()
                print(f"STEP 5: Model successfully registered in DB with SHA: {model_sha}")
                
            except Exception as e:
                logger.error(f"FATAL: Failed to register model in local DB: {e}", exc_info=True)

            return {
                "status": "success",
                "message": f"Model trained and logged to MLflow.",
                "mlflow_run_id": run.info.run_id,
                "accuracy": accuracy,
                "model_name": f"{model_class_name}_{model_name}"
            }
            
    except Exception as e:
        # General handler for the entire function
        import traceback
        error_message = f"FATAL CRASH! Exception: {e.__class__.__name__}, Details: {str(e)}, Traceback: {traceback.format_exc()}"
        print(error_message)
        logger.error(f"Internal crash during training: {e.__class__.__name__}", exc_info=True)
        return {"status": "failed", "message": f"Internal crash during training: {e.__class__.__name__}. Check API logs."}

# END OF FILE