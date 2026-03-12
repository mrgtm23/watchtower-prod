FROM python:3.11-slim

WORKDIR /app

# system deps for psycopg2, libmagic
# system deps for psycopg2, libmagic
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        gcc \
        libmagic1 \
        curl \
        tk \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy code
COPY scripts /app/scripts

COPY backend/app /app/app

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH="/app:$PYTHONPATH"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]