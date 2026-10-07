"""Tests for the test harness itself.

If any of these fail, the rest of the suite's "offline, no side effects" claim is void,
so they are kept permanently rather than checked once by hand.
"""
import socket

import pytest
from pytest_socket import SocketBlockedError

import s3_utils
from backend import main, services


def test_network_is_blocked():
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("1.1.1.1", 443), timeout=1)


def test_telemetry_db_is_redirected(telemetry_db):
    assert services.TELEMETRY_DB_PATH == str(telemetry_db)
    assert services.TELEMETRY_DB_PATH != services.os.path.join(services.BASE_DIR, "telemetry.db")


def test_s3_is_unconfigured_and_client_refuses():
    assert not s3_utils.is_configured()
    with pytest.raises(AssertionError):
        s3_utils._client()


def test_client_does_not_run_lifespan(client):
    # Building the real engine is what lifespan does; if it had run, this would be set.
    assert "rag_engine" not in main.server_state
    assert client.get("/api/v1/health").json()["engine_ready"] is False


def test_qdrant_is_patched_where_main_looks_it_up(qdrant_up, client):
    client.get("/api/v1/health")
    assert qdrant_up.calls == 1
