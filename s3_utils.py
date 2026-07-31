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


def is_configured() -> bool:
    """True when uploads/downloads will actually happen rather than silently no-op.

    Reports the same _BUCKET the transfer functions check, so /health can't claim
    persistence is on while every call is quietly returning False.
    """
    return bool(_BUCKET)


# download_status() return values.
NOT_CONFIGURED = "not_configured"  # no AWS_S3_BUCKET; uploads are no-ops too, so nothing is at risk
OK = "ok"                          # object existed and is now at local_path
MISSING = "missing"                # bucket reachable, key genuinely absent -- safe to start fresh
ERROR = "error"                    # object may exist but we could not read it -- DO NOT overwrite it


def download_status(key: str, local_path: str) -> str:
    """Downloads s3://<bucket>/<key> to local_path, reporting *why* it failed.

    download_file() collapses "there is no history yet" and "there is history but we
    could not fetch it" into the same False. Callers that then upload their own state
    destroy the remote copy in the second case, silently and unrecoverably -- which is
    how the telemetry history was lost. Anything restoring state that it will later
    write back should use this and refuse to upload on ERROR.

    Downloads to a temp file and renames on success, so a transfer that dies midway
    can't leave a truncated file that later looks like a valid restore.
    """
    if not _BUCKET:
        return NOT_CONFIGURED

    client = _client()
    try:
        client.head_object(Bucket=_BUCKET, Key=key)
    except Exception as e:
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            # A missing *bucket* also 404s here, and a typo'd AWS_S3_BUCKET must not be
            # read as "no history yet" -- that hands the caller a green light to write
            # over the real bucket's data once the name is corrected. Only call it
            # MISSING once the bucket itself is confirmed reachable.
            try:
                client.head_bucket(Bucket=_BUCKET)
            except Exception as be:
                print(f"[s3_utils] Bucket unreachable ({be}); not treating '{key}' as absent.")
                return ERROR
            print(f"[s3_utils] No existing '{key}' in bucket; starting fresh.")
            return MISSING
        # 403 lands here too: a key we're not allowed to read may still exist.
        print(f"[s3_utils] Could not stat '{key}' ({e}); treating as unreadable.")
        return ERROR

    tmp_path = f"{local_path}.s3tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        client.download_file(_BUCKET, key, tmp_path)
        os.replace(tmp_path, local_path)
        return OK
    except Exception as e:
        print(f"[s3_utils] Failed to download '{key}' ({e}); local copy left untouched.")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return ERROR


def download_file(key: str, local_path: str) -> bool:
    """Downloads s3://<bucket>/<key> to local_path. Returns True on success.

    Kept for callers that only care whether a file arrived. If the caller will later
    upload over the same key, use download_status() instead -- see the note there.
    """
    return download_status(key, local_path) == OK


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
