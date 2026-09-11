from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_FILE = PROJECT_ROOT / "README.md"


def test_readme_exists():
    """Verify that the project documentation exists."""
    assert README_FILE.exists()


def test_readme_is_not_empty():
    """Verify that the README contains documentation."""
    content = README_FILE.read_text(encoding="utf-8")
    assert content.strip()


def test_readme_contains_project_title():
    """Verify that the README contains the expected project title."""
    content = README_FILE.read_text(encoding="utf-8")
    assert "Documentation, Testing & Code Review" in content


def test_readme_mentions_ai_ml_stack():
    """Verify that the approved AI/ML stack is documented."""
    content = README_FILE.read_text(encoding="utf-8")

    required_tools = [
        "CrewAI",
        "LangGraph",
        "MLflow",
        "Ragas",
        "MLOps",
    ]

    for tool in required_tools:
        assert tool in content
