FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DOCUTRUST_DATA_DIR=/app/.data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

RUN mkdir -p /app/.data
EXPOSE 8010
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
