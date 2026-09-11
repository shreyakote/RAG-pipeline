from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


DOCUMENTS_FOLDER = Path("documents")


def build_bm25_pipeline():
    document_store = InMemoryDocumentStore()

    converter = PyPDFToDocument()

    writer = DocumentWriter(
        document_store=document_store
    )

    retriever = InMemoryBM25Retriever(
        document_store=document_store
    )

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(
        "converter",
        converter
    )

    indexing_pipeline.add_component(
        "writer",
        writer
    )

    indexing_pipeline.connect(
        "converter.documents",
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

    retrieval_pipeline = Pipeline()

    retrieval_pipeline.add_component(
        "retriever",
        retriever
    )

    return retrieval_pipeline, document_store


if __name__ == "__main__":
    pipeline, document_store = build_bm25_pipeline()

    print(
        f"Indexed documents: {document_store.count_documents()}"
    )

    question = input("\nEnter your question: ")

    result = pipeline.run(
        {
            "retriever": {
                "query": question,
                "top_k": 3
            }
        }
    )

    documents = result["retriever"]["documents"]

    print("\nRetrieved documents:")
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