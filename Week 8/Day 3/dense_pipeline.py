from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


DOCUMENTS_FOLDER = Path("documents")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_dense_pipeline():
    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    converter = PyPDFToDocument()

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "converter",
        converter
    )

    indexing_pipeline.add_component(
        "document_embedder",
        document_embedder
    )

    indexing_pipeline.add_component(
        "writer",
        writer
    )

    indexing_pipeline.connect(
        "converter.documents",
        "document_embedder.documents"
    )

    indexing_pipeline.connect(
        "document_embedder.documents",
        "writer.documents"
    )

    pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in the documents folder."
        )

    indexing_pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )

    retrieval_pipeline = Pipeline()

    retrieval_pipeline.add_component(
        "text_embedder",
        text_embedder
    )

    retrieval_pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store
        )
    )

    retrieval_pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    return retrieval_pipeline, document_store


if __name__ == "__main__":
    pipeline, document_store = build_dense_pipeline()

    print(
        f"Indexed documents: {document_store.count_documents()}"
    )

    question = input("\nEnter your question: ")

    result = pipeline.run(
        {
            "text_embedder": {
                "text": question
            },
            "retriever": {
                "top_k": 3
            }
        }
    )

    documents = result["retriever"]["documents"]

    print("\nDense Retrieval Results")
    print("=" * 60)

    for index, document in enumerate(documents, start=1):
        print(f"\nResult {index}")
        print("-" * 60)

        print(
            "File:",
            document.meta.get("file_path", "Unknown")
        )

        print(
            "Score:",
            document.score
        )

        print("\nContent:")
        print(document.content[:1000])