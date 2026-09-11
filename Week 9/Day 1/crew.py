import mlflow
from crewai import LLM, Agent, Crew, Process, Task

from web_search import WebSearchTool

TOPIC = "Artificial Intelligence and Retrieval-Augmented Generation"


def get_llm():
    """Create the local Ollama LLM."""

    return LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
    )


def create_agents():
    """Create Researcher, Writer and Reviewer agents."""

    llm = get_llm()

    researcher = Agent(
        role="Researcher",
        goal="Research the topic and provide accurate information.",
        backstory=(
            "You are an experienced AI researcher who collects "
            "important facts and explains technical topics clearly."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer",
        goal="Write a clear and well-structured article.",
        backstory=(
            "You are a technical writer who turns research "
            "information into simple and readable content."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Check the article for accuracy, clarity and quality.",
        backstory=(
            "You are a careful technical reviewer who checks "
            "content for errors, unsupported claims and poor structure."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer, topic, use_web=False):
    """Create the three tasks."""

    web_instruction = ""

    if use_web:
        web_instruction = """
Use the Web Search tool to find current information about the topic.
Include useful information from the search results in your research.
"""

    research_task = Task(
        description=f"""
Research the following topic:

{topic}

{web_instruction}

Identify important concepts, applications, benefits,
limitations and useful facts.

Do not invent information.
""",
        expected_output=(
            "A clear research report containing important facts, "
            "concepts, applications, benefits and limitations."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=f"""
Write a simple technical article about:

{topic}

Use the Researcher's findings.

The article must contain:

1. Introduction
2. Main concepts
3. Applications
4. Benefits
5. Limitations
6. Conclusion

Use clear and easy-to-understand language.
""",
        expected_output=(
            "A clear technical article with introduction, "
            "main concepts, applications, benefits, limitations "
            "and conclusion."
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

Then provide the improved final article.
""",
        expected_output=(
            "A corrected and improved final article."
        ),
        agent=reviewer,
        context=[writing_task],
    )

    return research_task, writing_task, review_task


def run_crew(topic, use_web=False):
    """Run the three-agent CrewAI workflow."""

    researcher, writer, reviewer = create_agents()

    if use_web:
        researcher.tools = [WebSearchTool()]

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer,
        topic,
        use_web,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    run_name = "web_research_crew" if use_web else "basic_research_crew"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("topic", topic)
        mlflow.log_param("agents", 3)
        mlflow.log_param("web_search", use_web)

        result = crew.kickoff()

    return result


def main():
    """Run both CrewAI experiments."""

    print("=" * 60)
    print("CREWAI MULTI-AGENT RESEARCH CREW")
    print("=" * 60)

    print("\nRunning basic crew without web search...\n")

    basic_result = run_crew(
        TOPIC,
        use_web=False,
    )

    with open(
        "basic_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(basic_result))

    print("\nBasic crew completed.")
    print("Saved: basic_output.txt")

    print("\n" + "=" * 60)
    print("Running crew with web search...")
    print("=" * 60)

    web_result = run_crew(
        TOPIC,
        use_web=True,
    )

    with open(
        "web_output.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(web_result))

    print("\nWeb-enabled crew completed.")
    print("Saved: web_output.txt")


if __name__ == "__main__":
    main()