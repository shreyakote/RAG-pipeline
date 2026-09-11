import asyncio
import csv
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_pipeline import ask_question, create_llm, create_vectorstore


load_dotenv()


def load_qa_pairs():
    with open("qa_pairs.json", "r", encoding="utf-8") as file:
        return json.load(file)


def get_score(result):
    if hasattr(result, "value"):
        return float(result.value)

    return float(result)


async def evaluate_all(samples, metrics):
    faithfulness = metrics["faithfulness"]
    answer_relevancy = metrics["answer_relevancy"]
    context_precision = metrics["context_precision"]
    context_recall = metrics["context_recall"]

    results = []

    for number, sample in enumerate(samples, start=1):

        print(f"\nEvaluating question {number}/10...")

        question = sample["user_input"]
        response = sample["response"]
        contexts = sample["retrieved_contexts"]
        reference = sample["reference"]

        result = {
            "user_input": question,
            "response": response,
            "reference": reference,
        }

        # Faithfulness
        try:
            score = await faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            )
            score = get_score(score)
        except Exception as error:
            print("Faithfulness error:", error)
            score = 0.0

        result["faithfulness"] = score
        print(f"Faithfulness       : {score:.4f}")

        # Answer Relevancy
        try:
            score = await answer_relevancy.ascore(
                user_input=question,
                response=response,
            )
            score = get_score(score)
        except Exception as error:
            print("Answer relevancy error:", error)
            score = 0.0

        result["answer_relevancy"] = score
        print(f"Answer Relevancy   : {score:.4f}")

        # Context Precision
        try:
            score = await context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            )
            score = get_score(score)
        except Exception as error:
            print("Context precision error:", error)
            score = 0.0

        result["context_precision"] = score
        print(f"Context Precision  : {score:.4f}")

        # Context Recall
        try:
            score = await context_recall.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference,
            )
            score = get_score(score)
        except Exception as error:
            print("Context recall error:", error)
            score = 0.0

        result["context_recall"] = score
        print(f"Context Recall     : {score:.4f}")

        results.append(result)

    return results


def create_samples():

    print("\nCreating ChromaDB vector store...")

    vectorstore = create_vectorstore()

    print("\nCreating Groq LLM...")

    llm = create_llm()

    qa_pairs = load_qa_pairs()

    samples = []

    print("\nRunning RAG for 10 questions...")
    print("-" * 60)

    for number, item in enumerate(qa_pairs, start=1):

        question = item["question"]

        result = ask_question(
            question,
            vectorstore,
            llm,
        )

        samples.append(
            {
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": item["answer"],
            }
        )

        print(f"Completed {number}/10: {question}")

    return samples


async def async_main():

    print("=" * 60)
    print("RAGAS BASELINE EVALUATION")
    print("=" * 60)

    samples = create_samples()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env"
        )

    print("\nCreating Groq evaluator client...")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print("Creating Ragas evaluator...")

    evaluator_llm = llm_factory(
        "openai/gpt-oss-20b",
        client=client,
    )

    print("Creating Ragas embeddings...")

    evaluator_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Ragas metrics...")

    metrics = {
        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),

        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        "context_precision": ContextPrecision(
            llm=evaluator_llm
        ),

        "context_recall": ContextRecall(
            llm=evaluator_llm
        ),
    }

    print("\nStarting Ragas evaluation...")
    print("This may take several minutes.")
    print("-" * 60)

    detailed_results = await evaluate_all(
        samples,
        metrics,
    )

    # ======================================================
    # AVERAGES
    # ======================================================

    print("\n")
    print("=" * 60)
    print("AVERAGE RAGAS SCORES")
    print("=" * 60)

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    averages = {}

    for metric in metric_names:

        values = [
            row[metric]
            for row in detailed_results
        ]

        average = sum(values) / len(values)

        averages[metric] = average

        print(
            f"{metric:20s}: {average:.4f}"
        )

    # ======================================================
    # LOWEST METRIC
    # ======================================================

    lowest_metric = min(
        averages,
        key=averages.get,
    )

    print("\n")
    print("=" * 60)
    print("LOWEST-SCORING METRIC")
    print("=" * 60)

    print(
        f"{lowest_metric}: "
        f"{averages[lowest_metric]:.4f}"
    )

    # ======================================================
    # SAVE CSV
    # ======================================================

    with open(
        "baseline_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "user_input",
            "response",
            "reference",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(detailed_results)

    # ======================================================
    # SAVE JSON
    # ======================================================

    summary = {
        "faithfulness": averages["faithfulness"],
        "answer_relevancy": averages["answer_relevancy"],
        "context_precision": averages["context_precision"],
        "context_recall": averages["context_recall"],
        "lowest_metric": lowest_metric,
        "lowest_score": averages[lowest_metric],
    }

    with open(
        "baseline_scores.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n")
    print("=" * 60)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 60)

    print("\nCreated:")
    print("  1. baseline_results.csv")
    print("  2. baseline_scores.json")

    print("\nNext task:")
    print(
        f"Optimize the lowest metric: {lowest_metric}"
    )


if __name__ == "__main__":
    asyncio.run(async_main())