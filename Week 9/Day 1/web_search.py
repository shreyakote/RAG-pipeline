from crewai.tools import BaseTool
from duckduckgo_search import DDGS


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for information about a topic."

    def _run(self, query: str) -> str:
        """Search the web and return search results."""

        try:
            results = DDGS().text(
                query,
                max_results=5,
            )

            if not results:
                return "No search results found."

            output = []

            for result in results:
                output.append(
                    f"Title: {result.get('title', '')}\n"
                    f"Information: {result.get('body', '')}\n"
                    f"Source: {result.get('href', '')}"
                )

            return "\n\n".join(output)

        except Exception as error:  # noqa: BLE001
            return f"Web search failed: {error}"