"""MinIO client helper for downloading model and video files."""

import os
import time
from minio import Minio
from minio.error import S3Error

from urllib.parse import urlparse


def get_minio_client():
    """Create and return a MinIO client from environment variables."""
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    # Minio() expects bare host:port; strip scheme if a full URL was provided
    parsed = urlparse(endpoint)
    if parsed.scheme in ("http", "https"):
        endpoint = parsed.netloc or parsed.path
        if parsed.scheme == "https":
            secure = True

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def download_file(
    bucket: str,
    object_name: str,
    local_path: str,
    max_retries: int = 5,
    retry_delay: int = 3,
) -> str:
    """
    Download a file from MinIO to a local path.

    Args:
        bucket: MinIO bucket name
        object_name: Object key/path in the bucket
        local_path: Local filesystem path to save the file
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries

    Returns:
        The local path where the file was saved

    Raises:
        S3Error: If download fails after all retries
    """
    client = get_minio_client()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(
                f"Downloading {bucket}/{object_name} to {local_path} (attempt {attempt + 1}/{max_retries})"
            )
            client.fget_object(bucket, object_name, local_path)
            print(f"Successfully downloaded {bucket}/{object_name}")
            return local_path
        except S3Error as e:
            if attempt < max_retries - 1:
                print(f"Download failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Download failed after {max_retries} attempts: {e}")
                raise
