"""Tavily web search tool for use with Claude tool_use."""

import os

import httpx

_API_KEY = os.environ.get("TAVILY_API_KEY")

TOOL_DEFINITION = {
    "name": "web_search",
    "description": (
        "Search the web for up-to-date information about MuleSoft, Anypoint Platform, "
        "DataWeave, connectors, and related Salesforce integration topics. "
        "Use this to find official documentation, Trailhead content, and authoritative resources."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10). Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def execute(query: str, num_results: int = 5) -> str:
    """Call Tavily Search API and return formatted results."""
    if not _API_KEY:
        return "ERROR: TAVILY_API_KEY environment variable must be set to use web search."

    num_results = max(1, min(10, num_results))
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": _API_KEY,
                "query": query,
                "max_results": num_results,
                "search_depth": "basic",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return f"Search request failed: {exc}"

    results = data.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, item in enumerate(results, 1):
        title = item.get("title", "No title")
        url = item.get("url", "")
        content = item.get("content", "No description available.").replace("\n", " ")
        lines.append(f"{i}. **{title}**\n   URL: {url}\n   {content}\n")

    return "\n".join(lines)
