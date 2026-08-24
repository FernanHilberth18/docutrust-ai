from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.store import DocumentStore


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(main_module, "store", DocumentStore(tmp_path / "data"))
    with TestClient(main_module.app) as test_client:
        yield test_client
