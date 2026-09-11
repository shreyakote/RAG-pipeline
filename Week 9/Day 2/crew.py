import mlflow
from crewai import Agent, Crew, LLM, Process, Task

from code_execution import CodeExecutionTool
from web_search import WebSearchTool


TOPIC = "Artificial Intelligence and Retrieval-Augmented Generation"


def get_llm():
    """Create the local Ollama LLM."""
    return LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
    )


def create_agents():
    """Create the Researcher, Writer and Reviewer agents."""

    llm = get_llm()

    researcher = Agent(
        role="Researcher",
        goal="Find accurate and useful information about the research topic.",
        backstory=(
            "You are an experienced AI researcher. "
            "You collect reliable information and identify "
            "important facts, applications, benefits and limitations."
        ),
        llm=llm,
        tools=[WebSearchTool(), CodeExecutionTool()],
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer",
        goal="Turn research findings into a clear technical article.",
        backstory=(
            "You are a technical writer who explains complex "
            "AI concepts in simple and organized language."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Review the article and improve its accuracy and quality.",
        backstory=(
            "You are a careful technical reviewer. "
            "You check accuracy, clarity, completeness, grammar "
            "and unsupported claims."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer, topic):
    """Create the research, writing and review tasks."""

    research_task = Task(
        description=f"""
Research the following topic:

{topic}

Use the Web Search tool to find useful and current information.

Use the Code Calculator tool if numerical calculations are needed.

Identify:

1. Important concepts
2. Applications
3. Benefits
4. Limitations
5. Useful facts

Do not invent information.
""",
        expected_output=(
            "A factual research report containing important concepts, "
            "applications, benefits, limitations and useful facts."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=f"""
Write a clear technical article about:

{topic}

Use the Researcher's findings.

Include:

1. Introduction
2. Main concepts
3. Applications
4. Benefits
5. Limitations
6. Conclusion

Use simple and understandable language.
""",
        expected_output=(
            "A well-structured technical article about the research topic."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description="""
Review the article created by the Writer.

Check:

- Accuracy
- Clarity
- Completeness
- Structure
- Unsupported claims
- Grammar
- Repetition

Provide the corrected final article.
""",
        expected_output="An improved and corrected final article.",
        agent=reviewer,
        context=[writing_task],
    )

    return research_task, writing_task, review_task


def run_crew(topic):
    """Run the multi-agent CrewAI workflow."""

    researcher, writer, reviewer = create_agents()

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer,
        topic,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    with mlflow.start_run(run_name="crewai_tools_research"):
        mlflow.log_param("topic", topic)
        mlflow.log_param("agents", 3)
        mlflow.log_param("web_search", True)
        mlflow.log_param("code_execution", True)

        result = crew.kickoff()

    return result


def main():
    """Run the research crew."""

    print("=" * 60)
    print("CREWAI TOOLS - WEB SEARCH + CODE EXECUTION")
    print("=" * 60)

    result = run_crew(TOPIC)

    with open(
        "final_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(result))

    print("\nCrew completed successfully.")
    print("Saved: final_output.txt")


if __name__ == "__main__":
    main()