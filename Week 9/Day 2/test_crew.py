from crew import create_agents, create_tasks


def test_three_agents():
    """Check that the three agents are created."""

    researcher, writer, reviewer = create_agents()

    assert researcher.role == "Researcher"
    assert writer.role == "Writer"
    assert reviewer.role == "Reviewer"


def test_three_tasks():
    """Check that the three tasks are created."""

    researcher, writer, reviewer = create_agents()

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer,
        "Artificial Intelligence",
    )

    assert research_task.agent == researcher
    assert writing_task.agent == writer
    assert review_task.agent == reviewer


def test_code_execution_tool():
    """Check that the code execution tool calculates correctly."""

    from code_execution import CodeExecutionTool

    tool = CodeExecutionTool()

    result = tool._run("10 + 20 * 2")

    assert result == "50"


def test_web_search_tool():
    """Check that the web search tool can be created."""

    from web_search import WebSearchTool

    tool = WebSearchTool()

    assert tool.name == "Web Search"