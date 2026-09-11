from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import CHROMA_PATH, EMBEDDING_MODEL


def create_vector_store():
    """Create the local ChromaDB vector store."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        collection_name="research_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


def load_documents():
    """Load the local research document."""
    file_path = Path("data/research.txt")
    text = file_path.read_text(encoding="utf-8")

    return [Document(page_content=text)]


def add_documents():
    """Add research documents to ChromaDB."""
    vector_store = create_vector_store()
    documents = load_documents()

    vector_store.add_documents(documents)

    return vector_store


def search_documents(question: str, k: int = 3):
    """Search ChromaDB for relevant research information."""
    vector_store = create_vector_store()

    return vector_store.similarity_search(question, k=k)
