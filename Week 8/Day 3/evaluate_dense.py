from dense_pipeline import build_dense_pipeline


QUESTIONS = [
    "What is machine learning?",
    "What are the different types of machine learning?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is classification?",
    "What is regression?",
    "What is cybersecurity?",
    "What are common cybersecurity threats?",
    "What is phishing?",
    "What are the basic principles of cybersecurity?",
]


def evaluate_dense():
    pipeline, document_store = build_dense_pipeline()

    print("=" * 70)
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Indexed documents: {document_store.count_documents()}")
    print(f"Total questions: {len(QUESTIONS)}")

    for number, question in enumerate(QUESTIONS, start=1):
        result = pipeline.run(
            {
                "text_embedder": {
                    "text": question
                },
                "retriever": {
                    "top_k": 3
                },
            }
        )

        documents = result["retriever"]["documents"]

        print("\n" + "=" * 70)
        print(f"QUESTION {number}")
        print("=" * 70)
        print(question)

        for rank, document in enumerate(documents, start=1):
            print(f"\nResult {rank}")
            print("-" * 40)

            print(
                "File:",
                document.meta.get("file_path", "Unknown")
            )

            print("Score:", document.score)

            print("Content:")
            print(
                document.content[:300].replace("\n", " ")
            )


if __name__ == "__main__":
    evaluate_dense()