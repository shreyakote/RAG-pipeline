from crew import create_agents, create_tasks


def test_three_agents():
    """Check that all three agents are created."""

    researcher, writer, reviewer = create_agents()

    assert researcher.role == "Researcher"
    assert writer.role == "Writer"
    assert reviewer.role == "Reviewer"


def test_three_tasks():
    """Check that all three tasks are created."""

    researcher, writer, reviewer = create_agents()

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer,
        "Generative AI",
    )

    assert research_task.agent == researcher
    assert writing_task.agent == writer
    assert review_task.agent == reviewer


def test_web_search_tool():
    """Check that the web search tool is available."""

    from web_search import WebSearchTool

    tool = WebSearchTool()

    assert tool.name == "Web Search"