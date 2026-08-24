from fastapi.testclient import TestClient

CONTENT = b"La politica de vacaciones permite 15 dias al ano. El permiso requiere aprobacion."


def test_upload_query_metrics_and_delete(client: TestClient) -> None:
    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("politica.txt", CONTENT, "text/plain")},
        data={"title": "Política de vacaciones"},
    )
    assert uploaded.status_code == 201
    document = uploaded.json()
    assert document["page_count"] == 1

    duplicate = client.post(
        "/api/v1/documents", files={"file": ("politica.txt", CONTENT, "text/plain")}
    )
    assert duplicate.status_code == 409

    query = client.post(
        "/api/v1/query", json={"question": "¿Cuántos días de vacaciones hay?", "top_k": 3}
    )
    assert query.status_code == 200
    assert query.json()["grounded"] is True
    assert query.json()["citations"][0]["document_id"] == document["id"]

    metrics = client.get("/api/v1/metrics").json()
    assert metrics == {"documents": 1, "chunks": 1, "queries": 1, "flagged_chunks": 0}
    assert client.delete(f"/api/v1/documents/{document['id']}").status_code == 204
    assert client.delete(f"/api/v1/documents/{document['id']}").status_code == 404


def test_health_and_invalid_upload(client: TestClient) -> None:
    assert client.get("/health").json()["mode"] == "local"
    invalid = client.post(
        "/api/v1/documents", files={"file": ("archivo.exe", b"bad", "application/octet-stream")}
    )
    assert invalid.status_code == 422
