import chromadb
import ollama
from sentence_transformers import SentenceTransformer


DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "eyedropshop_knowledge"
OLLAMA_MODEL = "qwen3.5:9b"
NUMBER_OF_RESULTS = 3


print("Loading the embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

chroma_client = chromadb.PersistentClient(path=DATABASE_FOLDER)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def retrieve_passages(question):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    number_of_results = min(NUMBER_OF_RESULTS, collection.count())

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=number_of_results,
    )

    passages = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(
            {
                "text": document,
                "source": metadata["source"],
                "distance": distance,
            }
        )

    return passages


def generate_answer(question, passages):
    evidence = "\n\n".join(
        f"[Source {index}: {passage['source']}]\n{passage['text']}"
        for index, passage in enumerate(passages, start=1)
    )

    system_prompt = """
You are the EyeDropShop Assistant.

Follow these rules:
1. Answer only from the provided evidence.
2. Do not use outside knowledge.
3. If the evidence does not contain the answer, say:
   "I don't have enough approved information to answer that."
4. Do not diagnose a medical condition.
5. Do not provide personalized medical treatment.
6. Cite sources using [Source 1], [Source 2], and so on.
7. Keep the answer clear and concise.
"""

    user_prompt = f"""
Question:
{question}

Approved evidence:
{evidence}

Write the answer and cite the supporting source.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.1,
        },
    )

    return response["message"]["content"]


question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a question.")
else:
    retrieved_passages = retrieve_passages(question)
    answer = generate_answer(question, retrieved_passages)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved sources:\n")

    for index, passage in enumerate(retrieved_passages, start=1):
        print(
            f"[Source {index}] {passage['source']} "
            f"(distance: {passage['distance']:.4f})"
        )