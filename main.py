import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI(title="Local Baseline RAG Service")

# Initialize local embedding model and Qdrant
print("Loading local embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(path="./qdrant_storage")

# Initialize a free local text-generation model compatible with the latest transformers library
print("Loading local LLM (HuggingFaceTB/SmolLM2-135M-Instruct)...")
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct", max_new_tokens=128)

COLLECTION_NAME = "rag_baseline"

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def answer_query(req: QueryRequest):
    try:
        # 1. Embed query
        query_vector = embedder.encode(req.query).tolist()

        # 2. Retrieve top chunks from vector DB using the updated API
        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        search_results = search_response.points

        # 3. Construct context
        context = "\n".join([hit.payload.get("text", "") for hit in search_results]) if search_results else "No relevant context found."
        sources = [hit.payload.get("source", "unknown") for hit in search_results]

        # 4. Generate answer using local LLM with a strict extraction prompt
        prompt = f"""Read the context below and extract the exact answer to the question. Keep your answer to one short sentence.

Context:
{context}

Question: {req.query}
Direct Answer:"""
        
        output = generator(prompt, max_new_tokens=128, return_full_text=False)
        answer_text = output[0]["generated_text"].strip()

        return {
            "answer": answer_text,
            "sources": list(set(sources))
        }
    except Exception as e:
        error_msg = traceback.format_exc()
        print("--- SERVER ERROR TRACEBACK ---")
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))