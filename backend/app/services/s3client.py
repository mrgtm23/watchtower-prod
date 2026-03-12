import boto3
import io
from backend.app.core.config import settings

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.MINIO_ENDPOINT,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        region_name="us-east-1",
        verify=False
    )

def ensure_bucket(bucket_name: str):
    s3 = get_s3_client()
    buckets = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
    if bucket_name not in buckets:
        s3.create_bucket(Bucket=bucket_name)

def upload_fileobj(fileobj, bucket_name: str, object_name: str):
    s3 = get_s3_client()
    ensure_bucket(bucket_name)
    s3.upload_fileobj(fileobj, bucket_name, object_name)
    return f"s3://{bucket_name}/{object_name}"

# --- NEW: Function definition required by drift_monitor.py ---
def download_fileobj(bucket_name, object_name, file_obj):
    """
    Downloads an object directly into a file-like object (e.g., io.BytesIO).
    """
    import boto3
    s3 = boto3.client(
        's3',
        endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    s3.download_fileobj(Bucket=bucket_name, Key=object_name, Fileobj=file_obj)

def download_to_file(bucket_name: str, object_name: str, dest_path: str):
    s3 = get_s3_client()
    with open(dest_path, "wb") as f:
        s3.download_fileobj(bucket_name, object_name, f)