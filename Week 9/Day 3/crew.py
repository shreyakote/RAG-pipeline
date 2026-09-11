import mlflow
from crewai import Agent, Crew, LLM, Process, Task

from web_search import WebSearchTool


TOPIC = "Generative AI and Large Language Models"


def get_llm():
    """Create the local Ollama LLM."""
    return LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
    )


def create_agents():
    """Create the three research agents."""

    llm = get_llm()

    researcher = Agent(
        role="Researcher",
        goal="Find accurate and relevant information about the research topic.",
        backstory=(
            "You are an AI researcher who investigates technical topics, "
            "collects important facts and identifies reliable information."
        ),
        llm=llm,
        tools=[WebSearchTool()],
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer",
        goal="Transform research findings into a clear and well-structured article.",
        backstory=(
            "You are a technical writer who explains complex technology "
            "in simple, organized and readable language."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Review the article and improve its accuracy and overall quality.",
        backstory=(
            "You are an experienced technical reviewer. "
            "You check facts, clarity, completeness, grammar and structure."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer, topic):
    """Create the research pipeline tasks."""

    research_task = Task(
        description=f"""
Research the following topic:

{topic}

Use the Web Search tool to collect current information.

Identify:

1. Definition
2. Important concepts
3. Applications
4. Benefits
5. Limitations
6. Recent developments

Do not invent information.
Use the web search results as supporting information.
""",
        expected_output=(
            "A detailed research report containing accurate information, "
            "applications, benefits, limitations and recent developments."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=f"""
Write a technical article about:

{topic}

Use the Researcher's findings.

The article must contain:

1. Introduction
2. Definition
3. Main concepts
4. Applications
5. Benefits
6. Limitations
7. Recent developments
8. Conclusion

Write in simple and clear language.
Do not add unsupported information.
""",
        expected_output=(
            "A clear, well-organized technical article based on the research."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description="""
Review the article produced by the Writer.

Check:

- Factual accuracy
- Clarity
- Completeness
- Organization
- Grammar
- Repetition
- Unsupported claims

Correct any problems you find and provide the final improved article.
""",
        expected_output=(
            "A corrected, accurate and polished final research article."
        ),
        agent=reviewer,
        context=[writing_task],
    )

    return research_task, writing_task, review_task


def run_crew(topic):
    """Run the multi-agent research pipeline."""

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

    with mlflow.start_run(run_name="research_crew_day3"):
        mlflow.log_param("topic", topic)
        mlflow.log_param("agents", 3)
        mlflow.log_param("web_search", True)

        result = crew.kickoff()

    return result


def main():
    """Run the research crew and save its output."""

    print("=" * 60)
    print("WEEK 9 DAY 3 - BUILDING A RESEARCH CREW")
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