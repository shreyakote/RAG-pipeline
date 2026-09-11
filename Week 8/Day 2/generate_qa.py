import json

from rag_pipeline import ask_question, create_llm, create_vectorstore


QUESTIONS = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is deep learning?",
    "What is natural language processing?",
    "What is computer vision?",
    "What is generative AI?",
    "What are large language models?",
    "What is retrieval augmented generation?",
    "What is ChromaDB?",
    "What is MLOps?",
]


def main():
    print("=" * 60)
    print("GENERATING 10 Q&A PAIRS")
    print("=" * 60)

    vectorstore = create_vectorstore()
    llm = create_llm()

    qa_pairs = []

    for number, question in enumerate(QUESTIONS, start=1):

        result = ask_question(
            question,
            vectorstore,
            llm,
        )

        answer = result["answer"].strip()

        qa_pairs.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        print(f"\nQ{number}: {question}")
        print(f"A{number}: {answer}")

    with open(
        "qa_pairs.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            qa_pairs,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("10 Q&A pairs saved to qa_pairs.json")
    print("=" * 60)


if __name__ == "__main__":
    main()