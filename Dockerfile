# HulioMed - FastAPI backend container (educational prototype deployment).
# The vector store is rebuilt at container start from the committed
# data/hulio_embeddings.json (no API key needed for the rebuild).
# MISTRAL_API_KEY must be provided as a secret/environment variable at runtime.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (sources + packaged data files).
COPY . .

# Start: (re)build the ChromaDB collection from the committed embeddings, then
# serve the API. Honors the PORT variable used by platforms like Render,
# defaulting to 8000 for plain `docker run`.
CMD ["sh", "-c", "python src/retrieval/store_vectors.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]