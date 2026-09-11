import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/dataset.txt"
CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4


# --------------------------------------------------
# Load documents
# --------------------------------------------------

def load_documents():
    """Load documents from the dataset."""

    loader = TextLoader(
        DATA_PATH,
        encoding="utf-8",
    )

    documents = loader.load()

    print(f"Documents loaded: {len(documents)}")

    return documents


# --------------------------------------------------
# Create vector store
# --------------------------------------------------

def create_vectorstore():
    """Create ChromaDB vector store."""

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="ragas_rag",
    )

    print("ChromaDB vector store created.")

    return vectorstore


# --------------------------------------------------
# Create Groq LLM
# --------------------------------------------------

def create_llm():
    """Create Groq LLM."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found in .env"
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )

    return llm


# --------------------------------------------------
# Ask question
# --------------------------------------------------

def ask_question(question, vectorstore, llm):
    """Retrieve relevant documents and answer the question."""

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    documents = retriever.invoke(question)

    contexts = [
        document.page_content
        for document in documents
    ]

    context = "\n\n".join(contexts)

    prompt = f"""
You are a helpful RAG question-answering assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 50)
    print("RAG PIPELINE")
    print("=" * 50)

    vectorstore = create_vectorstore()

    llm = create_llm()

    question = "What is artificial intelligence?"

    result = ask_question(
        question,
        vectorstore,
        llm,
    )

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CONTEXT:")

    for index, context in enumerate(
        result["contexts"],
        start=1,
    ):
        print(f"\nContext {index}:")
        print(context)


if __name__ == "__main__":
    main()