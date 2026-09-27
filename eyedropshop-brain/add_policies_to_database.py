from pathlib import Path
import re

from governance_registry import (
    get_governance_record,
    governance_metadata,
    record_is_indexable,
)

import chromadb
from sentence_transformers import SentenceTransformer
from ingestion_validator import validate_policies


DATABASE_FOLDER = "chroma_products_db"
COLLECTION_NAME = "eyedropshop_products"
EMBEDDING_MODEL = "BAAI/bge-m3"

POLICY_FILES = [
    {
        "path": Path(
            "knowledge/converted/policies/"
            "shipping_ca_en.txt"
        ),
        "language": "en",
        "policy_type": "shipping",
        "title": "Shipping Policy",
    },
    {
        "path": Path(
            "knowledge/converted/policies/"
            "shipping_ca_fr.txt"
        ),
        "language": "fr",
        "policy_type": "shipping",
        "title": "Politique d’expédition",
    },
    {
        "path": Path(
            "knowledge/converted/policies/"
            "return_ca_en.txt"
        ),
        "language": "en",
        "policy_type": "returns",
        "title": "Return Policy",
    },
    {
        "path": Path(
            "knowledge/converted/policies/"
            "return_ca_fr.txt"
        ),
        "language": "fr",
        "policy_type": "returns",
        "title": "Politique de retour",
    },
]


def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text, max_words=180):
    lines = [
        clean_text(line)
        for line in text.splitlines()
        if clean_text(line)
    ]

    chunks = []
    current_chunk = []
    current_words = 0

    for line in lines:
        line_words = len(line.split())

        if current_chunk and (
            current_words + line_words > max_words
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(line)
        current_words += line_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

print("Running policy ingestion validation...")

if not validate_policies():
    raise SystemExit(
        "Policy ingestion validation failed. "
        "Policy ingestion stopped."
    )

documents = []
document_ids = []
metadatas = []

for policy in POLICY_FILES:
    source = policy["path"].name

    governance_record = get_governance_record(
        source=source,
        content_type="policy",
        language=policy["language"],
        country="CA",
    )

    if not record_is_indexable(
        governance_record
    ):
        print(
            "[governance] skipping "
            f"non-indexable policy: {source}"
        )
        continue

    source_governance = governance_metadata(
        governance_record
    )

    text = policy["path"].read_text(
        encoding="utf-8",
        errors="replace",
    )

    chunks = create_chunks(text)

    for chunk_number, chunk in enumerate(chunks):
        documents.append(chunk)

        document_ids.append(
            f"policy-{policy['policy_type']}-"
            f"{policy['language']}-{chunk_number}"
        )

        metadata = {
            "content_type": "policy",
            "policy_type": policy["policy_type"],
            "title": policy["title"],
            "language": policy["language"],
            "country": "CA",
            "section": policy["policy_type"],
            "source": source,
            "priority": 1,
            "url": "",
        }

        metadata.update(
            source_governance
        )

        metadatas.append(
            metadata
        )


print("Loading BGE-M3...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Creating embeddings for {len(documents)} policy passages...")

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

print("Policies added successfully.")
print("Policy passages added:", len(documents))
print("Total database passages:", collection.count())