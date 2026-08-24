import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentParseError(ValueError):
    pass


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_document(filename: str, content: bytes) -> tuple[list[str], str]:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise DocumentParseError("Formato no permitido. Usa PDF, TXT o Markdown.")
    if not content:
        raise DocumentParseError("El archivo está vacío.")

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentParseError("El PDF está cifrado y no puede procesarse.")
            pages = [normalize_text(page.extract_text() or "") for page in reader.pages]
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("No fue posible leer el PDF.") from exc
        mime_type = "application/pdf"
    else:
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("latin-1")
        pages = [normalize_text(decoded)]
        mime_type = "text/markdown" if extension == ".md" else "text/plain"

    if not any(pages):
        raise DocumentParseError("No se encontró texto extraíble en el documento.")
    return pages, mime_type


def chunk_pages(pages: list[str], size: int = 140, overlap: int = 30) -> list[dict]:
    chunks: list[dict] = []
    for page_number, page_text in enumerate(pages, start=1):
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({"page": page_number, "text": " ".join(words[start:end])})
            if end == len(words):
                break
            start = end - overlap
    return chunks
