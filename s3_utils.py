"""Thin S3 wrapper for chunks.pkl / tracker file / telemetry backups.

Both backend/services.py and build_pipeline_NM.py import this. All functions
are no-ops when AWS_S3_BUCKET isn't set, so local dev without an AWS account
keeps working exactly as it does today.
"""
import os

_BUCKET = os.environ.get("AWS_S3_BUCKET")


def _client():
    import boto3
    return boto3.client("s3")


def download_file(key: str, local_path: str) -> bool:
    """Downloads s3://<bucket>/<key> to local_path. Returns True on success."""
    if not _BUCKET:
        return False
    try:
        _client().download_file(_BUCKET, key, local_path)
        return True
    except Exception as e:
        print(f"[s3_utils] Skipping download of '{key}' ({e})")
        return False


def upload_file(local_path: str, key: str) -> bool:
    """Uploads local_path to s3://<bucket>/<key>. Returns True on success."""
    if not _BUCKET:
        return False
    if not os.path.exists(local_path):
        return False
    try:
        _client().upload_file(local_path, _BUCKET, key)
        return True
    except Exception as e:
        print(f"[s3_utils] Skipping upload of '{key}' ({e})")
        return False
