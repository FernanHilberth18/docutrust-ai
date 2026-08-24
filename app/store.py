import hashlib
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.parsers import chunk_pages
from app.security import contains_untrusted_instruction


class DuplicateDocumentError(ValueError):
    pass


class DocumentStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index_path = data_dir / "index.json"
        self.files_dir = data_dir / "documents"
        self._lock = threading.RLock()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"documents": [], "chunks": [], "queries": 0}

    def _save(self) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.index_path)

    @property
    def documents(self) -> list[dict]:
        with self._lock:
            return [document.copy() for document in self._data["documents"]]

    @property
    def chunks(self) -> list[dict]:
        with self._lock:
            return [chunk.copy() for chunk in self._data["chunks"]]

    def add_document(
        self,
        *,
        filename: str,
        title: str,
        content: bytes,
        pages: list[str],
        mime_type: str,
    ) -> dict:
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            if any(document["sha256"] == digest for document in self._data["documents"]):
                raise DuplicateDocumentError("Este documento ya fue ingresado.")
            document_id = uuid.uuid4().hex[:12]
            raw_chunks = chunk_pages(pages)
            chunks = []
            for position, raw_chunk in enumerate(raw_chunks, start=1):
                chunks.append(
                    {
                        "id": f"{document_id}-{position:04d}",
                        "document_id": document_id,
                        "document_title": title,
                        "page": raw_chunk["page"],
                        "text": raw_chunk["text"],
                        "untrusted": contains_untrusted_instruction(raw_chunk["text"]),
                    }
                )
            document = {
                "id": document_id,
                "title": title,
                "filename": Path(filename).name,
                "sha256": digest,
                "mime_type": mime_type,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "warning_count": sum(chunk["untrusted"] for chunk in chunks),
                "created_at": datetime.now(UTC).isoformat(),
            }
            stored_name = f"{document_id}{Path(filename).suffix.lower()}"
            (self.files_dir / stored_name).write_bytes(content)
            self._data["documents"].append(document)
            self._data["chunks"].extend(chunks)
            self._save()
            return document.copy()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            document = next(
                (item for item in self._data["documents"] if item["id"] == document_id), None
            )
            if document is None:
                return False
            self._data["documents"] = [
                item for item in self._data["documents"] if item["id"] != document_id
            ]
            self._data["chunks"] = [
                item for item in self._data["chunks"] if item["document_id"] != document_id
            ]
            stored_path = (
                self.files_dir / f"{document_id}{Path(document['filename']).suffix.lower()}"
            )
            stored_path.unlink(missing_ok=True)
            self._save()
            return True

    def record_query(self) -> None:
        with self._lock:
            self._data["queries"] += 1
            self._save()

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "documents": len(self._data["documents"]),
                "chunks": len(self._data["chunks"]),
                "queries": self._data["queries"],
                "flagged_chunks": sum(chunk["untrusted"] for chunk in self._data["chunks"]),
            }
