from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "WatchTower AI"
    DATABASE_URL: str
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str = "models"
    MINIO_BUCKET: str = "mlflow-artifacts"
    MLFLOW_URI: str = "http://mlflow:5000"
    MLFLOW_S3_ENDPOINT_URL: str = "http://minio:9000"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"

    class Config:
        env_file = ".env"

settings = Settings()
