from fastapi import FastAPI
from pydantic import BaseModel

from dense_pipeline import build_dense_pipeline


app = FastAPI(
    title="Haystack RAG API",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


pipeline, document_store = build_dense_pipeline()


@app.get("/")
def root():
    return {
        "message": "Haystack RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents": document_store.count_documents(),
    }


@app.post("/query")
def query(request: QueryRequest):
    result = pipeline.run(
        {
            "text_embedder": {
                "text": request.question
            },
            "retriever": {
                "top_k": 3
            },
        }
    )

    documents = result["retriever"]["documents"]

    results = []

    for document in documents:
        results.append(
            {
                "content": document.content[:1000],
                "score": document.score,
                "file": document.meta.get(
                    "file_path",
                    "Unknown",
                ),
            }
        )

    return {
        "question": request.question,
        "results": results,
    }