FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY datagen ./datagen
COPY evals ./evals

RUN python -m pip install --upgrade pip && \
    pip install -e .[datagen]

EXPOSE 8000

CMD ["uvicorn", "docextract.api.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
