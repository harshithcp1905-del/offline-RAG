import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Qdrant client in local storage mode (saved locally in a folder named 'qdrant_storage')
qdrant = QdrantClient(path="./qdrant_storage")
COLLECTION_NAME = "rag_baseline"

# Initialize local embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sample technical documents to ingest
documents = [
    {
        "id": 1,
        "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
        "source": "FastAPI Docs"
    },
    {
        "id": 2,
        "text": "Qdrant is a vector similarity search engine and vector database. It provides a production-ready service with a convenient API to store, search, and manage points with associated payloads.",
        "source": "Qdrant Docs"
    },
    {
        "id": 3,
        "text": "Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from an external knowledge base.",
        "source": "AI Engineering Handbook"
    },
    {
        "id": 4,
        "text": "Sentence-Transformers is a Python framework for state-of-the-art sentence, text and image embeddings, built on top of PyTorch and Transformers.",
        "source": "HuggingFace Docs"
    }
]

def recreate_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,  # dimension size for all-MiniLM-L6-v2
            distance=models.Distance.COSINE
        )
    )
    print(f"Created fresh collection: {COLLECTION_NAME}")

def ingest_data():
    recreate_collection()
    
    points = []
    for doc in documents:
        vector = embedder.encode(doc["text"]).tolist()
        
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=vector,
                payload={
                    "text": doc["text"],
                    "source": doc["source"]
                }
            )
        )
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Successfully ingested {len(documents)} documents into Qdrant local storage!")

if __name__ == "__main__":
    ingest_data()