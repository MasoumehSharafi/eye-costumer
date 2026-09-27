import chromadb
from sentence_transformers import SentenceTransformer


DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "eyedropshop_knowledge"

embedding_model = SentenceTransformer("BAAI/bge-m3")

chroma_client = chromadb.PersistentClient(
    path=DATABASE_FOLDER
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

question = input("Enter your question: ")

question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True,
).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3,
)

print("\nMost relevant passages:\n")

for passage, source, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"Source: {source['source']}")
    print(f"Distance: {distance:.4f}")
    print(f"Passage: {passage}")
    print("-" * 50)