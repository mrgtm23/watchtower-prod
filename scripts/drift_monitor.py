# scripts/drift_monitor.py (UPDATED)
import pandas as pd
import os
import logging
import json
import io
from typing import List, Union
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from backend.app.services.s3client import download_fileobj 
from backend.app.core.config import settings
import boto3


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# INFERENCE_LOG_PATH = "logs/inference.log"
REFERENCE_BASE_DIR = "scripts/reference_data"

def generate_drift_report(user_id: str, 
                          model_sha: str,
                          reference_s3_path: str,
                          feature_columns: List[str]) -> Union[str, dict]:
    """
    Generates the Evidently Data Drift report using reference data downloaded 
    from MinIO and the model-specific inference log.
    """
    if not feature_columns:
        return {"error": "Feature columns must be provided for drift analysis."}

    # user_reference_path = os.path.join(REFERENCE_BASE_DIR, user_id, "reference.csv")
    
    
    # if not os.path.exists(user_reference_path):
    #     return {"error": "Reference data not uploaded. Please use the 'Upload Reference Dataset' feature."}

    # --- 1. Load Reference Data ---
    try:
        # Extract the object name from the full S3 path
        object_name = reference_s3_path.split(f"s3://{settings.MINIO_BUCKET}/")[-1]
        file_obj = io.BytesIO()
        # Download data directly into the in-memory buffer
        download_fileobj(settings.MINIO_BUCKET, object_name, file_obj) 
        
        file_obj.seek(0)
        reference_df = pd.read_csv(file_obj)
        reference_df = reference_df[feature_columns]

        # reference_df = pd.read_csv(user_reference_path)
        # # Select only the features provided by the user
        # reference_df = reference_df[feature_columns] 
    except KeyError as e:
        return {"error": f"Missing column in reference data: {e}. Check feature names."}
    except Exception as e:
        return {"error": f"Error loading reference CSV: {e}"}

    # --- 2. Load Current (Inference) Data ---
    log_data = []
    log_prefix = f"logs/{model_sha}/"

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        
        # List all log files for this model SHA
        response = s3.list_objects_v2(
            Bucket=settings.MINIO_BUCKET, 
            Prefix=log_prefix
        )
        
        if 'Contents' not in response:
            print("Inference log is empty. Run model predictions before checking drift.")
            return {"error": "Inference log is empty. Run model predictions before checking drift."}

        # Iterate over all found log files and download/parse
        for obj in response['Contents']:
            file_obj = io.BytesIO()
            # download_fileobj(bucket_name, object_name, file_obj) is usually in s3client.py
            s3.download_fileobj(settings.MINIO_BUCKET, obj['Key'], file_obj)
            file_obj.seek(0)
            
            # Assuming each log entry is a single JSON line:
            log_line = file_obj.read().decode('utf-8').strip()
            if log_line:
                log_data.append(json.loads(log_line)['input'])
                
    except Exception as e:
        return {"error": f"Error reading or parsing inference logs from MinIO: {e}"}

    if not log_data:
        return {"error": "Inference log is empty after parsing."}

    # Convert logged inputs to a DataFrame using user-provided column names
    current_df = pd.DataFrame(log_data, columns=feature_columns)

    # current_data_path = os.path.join("logs", f"inference_{model_sha}.log") # Use isolated log
    
    # if not os.path.exists(current_data_path) or os.stat(current_data_path).st_size == 0:
    #     return {"error": "Inference log is empty. Run model predictions before checking drift."}

    # log_data = []
    # try:
    #     with open(current_data_path, 'r') as f:
    #         for line in f:
    #             # We assume the log line is a JSON object containing an 'input' array
    #             log_data.append(json.loads(line)['input'])
    # except Exception as e:
    #     return {"error": f"Error reading or parsing inference log: {e}"}

    # if not log_data:
    #     return {"error": "Inference log is empty after parsing."}

    # # Convert logged inputs to a DataFrame using user-provided column names
    # current_df = pd.DataFrame(log_data, columns=feature_columns)
    
    # --- 3. Run Evidently Report ---
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    data_drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=None
    )

    temp_report_path = f"/tmp/drift_report_{user_id}.html"

    data_drift_report.save_html(temp_report_path)

    try:
        with open(temp_report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        return {"error": f"Failed to read generated HTML file: {e}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_report_path):
            os.remove(temp_report_path)


    print(f"Drift report generated successfully. HTML content length: {len(html_content)}")

    # Return the HTML content directly
    return html_content

if __name__ == "__main__":
    # Remove main execution block for safety in production
    pass