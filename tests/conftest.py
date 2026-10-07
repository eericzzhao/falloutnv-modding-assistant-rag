"""Shared fixtures that keep the suite offline.

Every seam here exists because the obvious approach silently does nothing:

- TELEMETRY_DB_PATH is joined from BASE_DIR at import, not read from the environment,
  so it has to be patched as a module attribute. That works because every telemetry
  function re-reads the global on each call.
- s3_utils._BUCKET is snapshotted at import (and before load_dotenv runs), so setenv
  on AWS_S3_BUCKET has no effect. Patch the attribute.
- TestClient(app) built *without* `with` never runs lifespan, so the real engine --
  ~1.2 GB of model downloads plus a live Qdrant dial -- is never constructed.
- backend.main imports qdrant_status by value, so it must be patched on backend.main;
  patching backend.services leaves /health and the /query error path untouched.
"""
import pytest
from fastapi.testclient import TestClient

import s3_utils
from backend import main, services


@pytest.fixture(autouse=True)
def telemetry_db(tmp_path, monkeypatch):
    """Points telemetry at a throwaway DB and resets the module-level telemetry state."""
    db_path = tmp_path / "telemetry.db"
    monkeypatch.setattr(services, "TELEMETRY_DB_PATH", str(db_path))
    monkeypatch.setattr(services, "_telemetry_log_count", 0)
    monkeypatch.setattr(services, "_telemetry_restore", "not_attempted")
    monkeypatch.setattr(services, "_telemetry_upload_blocked", False)
    return db_path


@pytest.fixture(autouse=True)
def no_s3(monkeypatch):
    """Forces the unconfigured-S3 path and makes building a real boto3 client an error.

    The socket guard would also catch a stray AWS call, but this fails with a message
    that says what happened instead of a generic SocketBlockedError.
    """
    monkeypatch.setattr(s3_utils, "_BUCKET", None)

    def _refuse():
        raise AssertionError("test tried to build a real boto3 S3 client")

    monkeypatch.setattr(s3_utils, "_client", _refuse)


class QdrantStub:
    """Stands in for backend.main.qdrant_status. Unreachable unless a test says otherwise."""

    def __init__(self):
        self.reachable = False
        self.calls = 0

    def __call__(self, timeout: float = 5.0):
        self.calls += 1
        if self.reachable:
            return {"collection": services.QDRANT_COLLECTION, "reachable": True, "points": 1619}
        return {
            "collection": services.QDRANT_COLLECTION,
            "reachable": False,
            "error": "ResponseHandlingException: Unexpected Response: 503 b'no available server'",
        }


@pytest.fixture(autouse=True)
def qdrant(monkeypatch):
    stub = QdrantStub()
    monkeypatch.setattr(main, "qdrant_status", stub)
    return stub


@pytest.fixture
def qdrant_up(qdrant):
    qdrant.reachable = True
    return qdrant


@pytest.fixture
def qdrant_down(qdrant):
    qdrant.reachable = False
    return qdrant


class FakeEngine:
    """Records run_query calls; raises `error` instead of answering when it is set."""

    def __init__(self, answer: str = "Use NVTF instead."):
        self.answer = answer
        self.error = None
        self.calls = []

    def run_query(self, query: str, route: str = "query"):
        self.calls.append({"query": query, "route": route})
        if self.error is not None:
            raise self.error
        return {
            "answer": self.answer,
            "candidate_pool_size": 2,
            "candidates": [
                {"text": "chunk a", "source_file": "safe-modding.html"},
                {"text": "chunk b", "source_file": "changelog.html"},
            ],
            "selected_context": [
                {"text": "chunk a", "source_file": "safe-modding.html", "rerank_score": 0.9},
            ],
        }


@pytest.fixture
def engine():
    fake = FakeEngine()
    main.server_state["rag_engine"] = fake
    yield fake
    main.server_state.clear()


@pytest.fixture
def client():
    # Deliberately no `with`: entering the context would run lifespan and build the
    # real FalloutRAGEngine. Tests that need an engine request the `engine` fixture.
    main.server_state.clear()
    yield TestClient(main.app)
    main.server_state.clear()
