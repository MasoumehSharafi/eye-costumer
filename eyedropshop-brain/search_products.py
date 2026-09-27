import chromadb
from sentence_transformers import SentenceTransformer


DATABASE_FOLDER = "chroma_products_db"
COLLECTION_NAME = "eyedropshop_products"
EMBEDDING_MODEL = "BAAI/bge-m3"
NUMBER_OF_RESULTS = 5


print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


language = input(
    "Choose language (en/fr): "
).strip().lower()

if language not in {"en", "fr"}:
    raise ValueError("Language must be en or fr.")

question = input("Enter your question: ").strip()

question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True,
).tolist()

number_of_results = min(
    NUMBER_OF_RESULTS,
    collection.count(),
)

results = collection.query(
    query_embeddings=question_embedding,
    n_results=number_of_results,
    where={
        "language": language,
    },
)

print("\nMost relevant passages:\n")

for index, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ),
    start=1,
):
    title = (
        metadata["product_title"]
        or metadata["product_handle"]
    )

    print(f"Result {index}")
    print(f"Product: {title}")
    print(f"Language: {metadata['language']}")
    print(f"Section: {metadata['section']}")
    print(f"Distance: {distance:.4f}")
    print(f"Source: {metadata['source']}")
    print(f"Content: {document}")
    print("-" * 60)