import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.rag_pipeline import search_documents
from src.research_assistant import research

QUESTIONS = [
    "What is artificial intelligence?",
    "What is Retrieval-Augmented Generation?",
    "How does RAG work?",
    "What are vector databases?",
    "What is ChromaDB?",
    "What are embeddings?",
    "What is Ollama used for?",
    "What is a large language model?",
    "What is MLflow used for?",
    "What is Ragas used for?",
]


def create_dataset():
    """Create RAG evaluation samples."""

    samples = []

    for question in QUESTIONS:
        print(f"\nProcessing: {question}")

        documents = search_documents(question, k=3)

        contexts = [document.page_content for document in documents]

        result = research(question)

        samples.append(
            {
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": contexts,
            }
        )

    return samples


def main():
    """Run Ragas evaluation."""

    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    samples = create_dataset()

    print("\nEvaluation dataset created.")
    print(f"Number of questions: {len(samples)}")

    print("\nRagas evaluation requires an LLM judge.")
    print("Your local Ollama model will be used.")

    print("\nEvaluation samples are ready.")

    for index, sample in enumerate(samples, start=1):
        print(f"\nQuestion {index}: {sample['user_input']}")
        print(f"Retrieved contexts: {len(sample['retrieved_contexts'])}")
        print(f"Answer length: {len(sample['response'])} characters")

    print("\n" + "=" * 60)
    print("RAGAS DATASET CREATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
