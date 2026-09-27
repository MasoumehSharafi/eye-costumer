from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


KNOWLEDGE_FOLDER = Path("knowledge")
DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "eyedropshop_knowledge"


print("Loading the embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

chroma_client = chromadb.PersistentClient(path=DATABASE_FOLDER)

# Delete the old collection when rebuilding the test database.
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.create_collection(
    name=COLLECTION_NAME
)

documents = []
document_ids = []
metadata = []

for file_path in KNOWLEDGE_FOLDER.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        continue

    # Simple paragraph-based splitting for the first prototype.
    passages = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    for passage_number, passage in enumerate(passages):
        documents.append(passage)
        document_ids.append(
            f"{file_path.stem}-{passage_number}"
        )
        metadata.append(
            {
                "source": file_path.name,
                "passage_number": passage_number,
            }
        )

if not documents:
    raise ValueError("No text was found in the knowledge folder.")

print(f"Creating embeddings for {len(documents)} passages...")

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
).tolist()

collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata,
)

print("Knowledge database created successfully.")