# Documentation, Testing & Code Review

## Project Overview

This project demonstrates a production-oriented AI/ML workflow using the approved AI/ML 3M stack:

- CrewAI
- LangGraph
- MLflow
- Ragas
- MLOps

The purpose of this task is to document the project, validate the implementation through automated testing, and perform a basic code review to improve code quality and maintainability.

## Objectives

The main objectives are:

1. Document the AI/ML project clearly.
2. Maintain clean and readable source code.
3. Add automated tests for important functionality.
4. Run tests and verify that they pass.
5. Perform code-quality checks.
6. Review the implementation for common issues.
7. Commit and push the completed work to GitHub.

## AI/ML 3M Stack

### CrewAI

CrewAI can be used to coordinate multiple AI agents and assign tasks to specialized agents.

### LangGraph

LangGraph can be used to design structured and stateful AI workflows.

### MLflow

MLflow provides experiment tracking and helps monitor machine-learning experiments and model-related metrics.

### Ragas

Ragas is used for evaluating Retrieval-Augmented Generation (RAG) systems using metrics such as:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

### MLOps

MLOps practices are used to support:

- Testing
- Continuous integration
- Deployment
- Monitoring
- Reproducibility
- Code quality

## Documentation

The project documentation should provide enough information for another developer to understand the purpose, setup, testing process, and workflow.

## Testing

Automated tests are used to verify that the implemented functionality behaves as expected.

The test suite can be executed using:

```powershell
pytest -v