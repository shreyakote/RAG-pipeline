from typing import TypedDict

import mlflow
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.config import OLLAMA_MODEL
from src.crewai_agent import analyze_research
from src.rag_pipeline import search_documents


class ResearchState(TypedDict):
    question: str
    context: str
    answer: str
    summary: str


def retrieve_documents(state: ResearchState):
    """Retrieve relevant documents from ChromaDB."""

    documents = search_documents(state["question"], k=3)

    context = "\n\n".join(document.page_content for document in documents)

    return {"context": context}


def generate_answer(state: ResearchState):
    """Generate an answer using the local Ollama model."""

    prompt = f"""
You are a local AI research assistant.

Answer the question using only the provided context.

Question:
{state["question"]}

Context:
{state["context"]}

Give a clear and concise answer.
"""

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)

    return {"answer": response.content}


def create_crewai_summary(state: ResearchState):
    """Use CrewAI to summarize the retrieved research."""

    summary = analyze_research(state["context"])

    return {"summary": summary}


def build_research_graph():
    """Build the complete LangGraph workflow."""

    graph = StateGraph(ResearchState)

    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("generate", generate_answer)
    graph.add_node("summarize", create_crewai_summary)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def research(question: str) -> dict:
    """Run the complete local research assistant."""

    app = build_research_graph()

    with mlflow.start_run():
        mlflow.log_param("model", OLLAMA_MODEL)
        mlflow.log_param("retrieval_k", 3)
        mlflow.log_param("question", question)

        result = app.invoke(
            {
                "question": question,
                "context": "",
                "answer": "",
                "summary": "",
            }
        )

    return result


if __name__ == "__main__":
    question = input("Enter your research question: ")

    result = research(question)

    print("\nResearch Answer:")
    print(result["answer"])

    print("\nCrewAI Research Summary:")
    print(result["summary"])
