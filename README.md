# DocuTrust AI

[![CI](https://github.com/FernanHilberth18/docutrust-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/FernanHilberth18/docutrust-ai/actions/workflows/ci.yml)

Sistema RAG local que consulta PDF, TXT y Markdown y responde únicamente con evidencia recuperada, referencias verificables y un indicador de confianza.

## Por qué destaca en un portafolio

- No requiere enviar documentos a servicios externos.
- Combina BM25 y similitud vectorial con un índice determinista.
- Conserva página, fragmento, hash SHA-256 y fecha de ingestión.
- Rechaza preguntas sin evidencia suficiente en lugar de inventar respuestas.
- Detecta instrucciones sospechosas dentro de documentos y las marca como contenido no confiable.
- Incluye API, interfaz web, evaluación reproducible, pruebas, Docker y CI.

## Inicio rápido

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8010
```

Abre `http://127.0.0.1:8010`, carga un documento y formula una pregunta.

## API

- `POST /api/v1/documents`: ingesta un PDF, TXT o MD.
- `GET /api/v1/documents`: lista documentos y trazabilidad.
- `DELETE /api/v1/documents/{id}`: elimina documento y fragmentos.
- `POST /api/v1/query`: recupera evidencia y genera respuesta extractiva con citas.
- `GET /api/v1/metrics`: muestra documentos, fragmentos y consultas.
- `GET /health`: verifica el servicio.

Documentación OpenAPI: `http://127.0.0.1:8010/docs`.

## Calidad y evaluación

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m pytest --cov=app --cov-report=term-missing
.venv\Scripts\python scripts\evaluate.py
```

El conjunto `evaluation/questions.json` mide precisión de recuperación, rechazo correcto y presencia de citas.

## Privacidad

Los archivos e índices se almacenan bajo `.data/`, carpeta ignorada por Git. El proyecto usa recuperación local y no necesita credenciales de IA.
