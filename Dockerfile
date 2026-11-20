FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DATA_DIR=/tmp/data \
    TRACE_DIR=/tmp/data/traces \
    SEED_DIR=/app/sample_data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY sample_data ./sample_data

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
