import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_project_files_exist():
    """Check that the main project files exist."""
    assert Path("src/config.py").exists()
    assert Path("src/rag_pipeline.py").exists()
    assert Path("src/research_assistant.py").exists()


def test_config_values():
    """Check that local AI configuration exists."""
    from src.config import EMBEDDING_MODEL, OLLAMA_MODEL

    assert OLLAMA_MODEL
    assert EMBEDDING_MODEL
