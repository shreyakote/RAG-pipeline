from crewai import LLM, Agent, Crew, Task

from src.config import OLLAMA_MODEL


def create_research_agent():
    """Create a CrewAI research assistant using local Ollama."""

    local_llm = LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url="http://localhost:11434",
    )

    researcher = Agent(
        role="Research Assistant",
        goal="Analyze retrieved research information and provide a clear answer.",
        backstory=(
            "You are a research assistant who analyzes technical "
            "information and explains it clearly."
        ),
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher


def analyze_research(context: str) -> str:
    """Analyze retrieved RAG context using CrewAI."""

    researcher = create_research_agent()

    task = Task(
        description=f"""
Analyze the following research information.

Research information:
{context}

Provide a concise and accurate summary.
Do not add information that is not present in the research information.
""",
        expected_output="A clear and concise research summary.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()

    return str(result)


if __name__ == "__main__":
    sample_context = """
    Retrieval-Augmented Generation combines information retrieval
    with a language model. Relevant documents are retrieved first
    and then provided as context to the language model.
    """

    print("Running CrewAI with local Ollama...\n")

    answer = analyze_research(sample_context)

    print("\nCrewAI Research Summary:")
    print(answer)
