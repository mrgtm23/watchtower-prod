import mlflow
import json
from backend.app.core.config import settings

mlflow.set_tracking_uri(settings.MLFLOW_URI)

def log_model_registration(model_local_path: str, model_name: str, params: dict = None, metrics: dict = None):
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.MLFLOW_URI)
    exp_name = "watchtower_experiment"
    if client.get_experiment_by_name(exp_name) is None:
        client.create_experiment(exp_name, artifact_location='mlruns')
    with mlflow.start_run(run_name=f"register-{model_name}") as run:
        if params:
            for k,v in params.items():
                mlflow.log_param(k, v)
        if metrics:
            for k,v in metrics.items():
                mlflow.log_metric(k, v)
        mlflow.log_artifact(model_local_path, artifact_path="model")
        # 2. CRITICAL FIX: Register the model version using the artifact's run URI
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model" 
        
        mlflow.register_model(
            model_uri=model_uri,
            name=model_name # Use the file name as the registered model name
        )
        return run.info.run_id