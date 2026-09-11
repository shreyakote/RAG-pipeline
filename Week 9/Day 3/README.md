# Week 9 Day 3 — Building a Research Crew

## Objective

Build a multi-agent research pipeline using CrewAI.

## Agents

### Researcher

Finds accurate and relevant information using web search.

### Writer

Converts the research findings into a clear technical article.

### Reviewer

Checks the article for accuracy, clarity, completeness, grammar and unsupported claims.

## Pipeline

Researcher
    ↓
Web Search
    ↓
Writer
    ↓
Reviewer
    ↓
Final Article

## Web Search

The Researcher uses a DuckDuckGo web search tool to obtain current information.

This improves the research by providing information from real web sources instead of relying only on the local language model.

## MLflow

MLflow tracks the research crew experiment.

Tracked information includes:

- Research topic
- Number of agents
- Web search usage

## Approved AI/ML 3M Stack

- CrewAI — multi-agent orchestration
- LangGraph — workflow orchestration from previous work
- MLflow — experiment tracking
- Ragas — RAG evaluation from previous work
- MLOps — testing, linting and Git workflow

## Local AI

The project uses Ollama with:

llama3.2:3b

## Testing

Run:

```powershell
pytest -v