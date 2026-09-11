from haystack_pipeline import build_bm25_pipeline


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


def evaluate_bm25():
    pipeline, document_store = build_bm25_pipeline()

    print("=" * 70)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"\nIndexed documents: {document_store.count_documents()}")
    print(f"Total questions: {len(QUESTIONS)}")

    results = []

    for number, question in enumerate(QUESTIONS, start=1):
        result = pipeline.run(
            {
                "retriever": {
                    "query": question,
                    "top_k": 3,
                }
            }
        )

        documents = result["retriever"]["documents"]

        print("\n" + "=" * 70)
        print(f"QUESTION {number}")
        print("=" * 70)
        print(question)

        for rank, document in enumerate(documents, start=1):
            file_name = document.meta.get(
                "file_path",
                "Unknown",
            )

            print(f"\nResult {rank}")
            print(f"File: {file_name}")
            print(f"Score: {document.score}")
            print("Content:")
            print(document.content[:300].replace("\n", " "))

        results.append(
            {
                "question": question,
                "documents": documents,
            }
        )

    return results


if __name__ == "__main__":
    evaluate_bm25()