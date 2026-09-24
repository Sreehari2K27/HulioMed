"""Phase 7 - Store the chunks, their Mistral embeddings and metadata in ChromaDB.

Reads data/hulio_embeddings.json (created in Phase 6) and writes everything
into a persistent ChromaDB collection stored on disk in data/chroma_db/.

ChromaDB lets us search by MEANING: later, when the chatbot embeds a user's
question, ChromaDB finds the stored chunks whose vectors are closest.

Note on metadata: ChromaDB metadata values must be simple scalars (numbers or
strings), so the page list [3, 4] is stored as the string "3,4" plus integer
page_start / page_end for easy filtering later.
"""

import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "hulio_embeddings.json"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "hulio_monograph"
DISTANCE_SPACE = "cosine"  # measure similarity by angle between vectors


def main() -> None:
    import json

    if not INPUT_PATH.exists():
        sys.exit(f"Embeddings JSON not found: {INPUT_PATH}")

    with open(INPUT_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload["records"]

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_SPACE},
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for record in records:
        ids.append(record["chunk_id"])
        embeddings.append(record["embedding"])
        documents.append(record["text"])
        pages = record["pages"]
        metadatas.append(
            {
                "document": record["document"],
                "pages": ",".join(str(p) for p in pages),
                "page_start": min(pages),
                "page_end": max(pages),
            }
        )

    # Replace any previous build so the collection always matches the current data.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"Collection:            {COLLECTION_NAME}")
    print(f"Vectors stored:        {collection.count()}")
    print(f"Similarity measure:    {DISTANCE_SPACE}")
    print(f"ChromaDB persistence:  {DB_PATH}")


if __name__ == "__main__":
    main()