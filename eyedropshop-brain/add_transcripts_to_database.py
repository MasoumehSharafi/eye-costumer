from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from governance_registry import (
    get_governance_record,
    governance_metadata,
    record_is_indexable,
)
from ingestion_validator import validate_transcripts


DATABASE_FOLDER = "chroma_products_db"
COLLECTION_NAME = "eyedropshop_products"
EMBEDDING_MODEL = "BAAI/bge-m3"

TRANSCRIPT_DIR = Path(
    "knowledge/converted/transcripts"
)


def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(
    text,
    max_words=180,
    overlap_words=30,
):
    words = clean_text(text).split()

    chunks = []

    start = 0

    while start < len(words):
        end = min(
            start + max_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(
            0,
            end - overlap_words,
        )

    return chunks


print(
    "Running transcript ingestion validation..."
)

if not validate_transcripts():
    raise SystemExit(
        "Transcript ingestion validation failed. "
        "Transcript ingestion stopped."
    )


documents = []
document_ids = []
metadatas = []


transcript_files = sorted(
    TRANSCRIPT_DIR.glob("*.txt")
)


for path in transcript_files:
    source = path.name

    governance_record = (
        get_governance_record(
            source=source,
            content_type="transcript",
            language="en",
            country="CA",
        )
    )

    if not record_is_indexable(
        governance_record
    ):
        print(
            "[governance] skipping "
            f"non-indexable transcript: {source}"
        )
        continue

    source_governance = (
        governance_metadata(
            governance_record
        )
    )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    chunks = create_chunks(
        text
    )

    episode = (
        path.stem
        .replace(" Transcript", "")
        .strip()
    )

    for chunk_number, chunk in enumerate(
        chunks
    ):
        documents.append(
            chunk
        )

        episode_slug = (
            episode
            .lower()
            .replace(" ", "-")
        )

        document_ids.append(
            f"transcript-{episode_slug}-"
            f"{chunk_number}"
        )

        metadata = {
            "content_type": "transcript",
            "title": f"{episode} Transcript",
            "episode": episode,
            "language": "en",
            "country": "CA",
            "section": "education",
            "source": source,
            "priority": 2,
            "url": "",
        }

        metadata.update(
            source_governance
        )

        metadatas.append(
            metadata
        )


print("Loading BGE-M3...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    f"Creating embeddings for "
    f"{len(documents)} transcript passages..."
)

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True,
).tolist()


client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


collection.upsert(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)


print(
    "Transcripts added successfully."
)

print(
    "Transcript passages added:",
    len(documents),
)

print(
    "Total database passages:",
    collection.count(),
)