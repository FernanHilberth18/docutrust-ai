from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.parsers import DocumentParseError, parse_document
from app.retrieval import answer_question
from app.schemas import DocumentRead, MetricsResponse, QueryRequest, QueryResponse
from app.store import DocumentStore, DuplicateDocumentError

settings = get_settings()
store = DocumentStore(settings.docutrust_data_dir)
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="RAG local con recuperación híbrida, citas y controles de confianza.",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_app() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment, "mode": "local"}


@app.post(
    "/api/v1/documents",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(...), title: str | None = Form(default=None)
) -> dict:
    filename = Path(file.filename or "document").name
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El archivo supera el tamaño permitido.")
    try:
        pages, mime_type = parse_document(filename, content)
        document_title = title.strip() if title and title.strip() else Path(filename).stem
        return store.add_document(
            filename=filename,
            title=document_title[:160],
            content=content,
            pages=pages,
            mime_type=mime_type,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@app.get("/api/v1/documents", response_model=list[DocumentRead], tags=["documents"])
def list_documents() -> list[dict]:
    return sorted(store.documents, key=lambda item: item["created_at"], reverse=True)


@app.delete("/api/v1/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: str) -> None:
    if not store.delete_document(document_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Documento no encontrado")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["retrieval"])
def query(payload: QueryRequest) -> QueryResponse:
    store.record_query()
    return answer_question(
        payload.question,
        store.chunks,
        top_k=payload.top_k,
        min_confidence=settings.min_confidence,
        document_ids=payload.document_ids,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["metrics"])
def metrics() -> dict[str, int]:
    return store.metrics()
