"""Researcher Agent — finds up-to-date MuleSoft information via web search."""

from __future__ import annotations

import anthropic

from config import MODEL, thinking_param, output_config_param
from tools import google_search
from agents.base import run_with_tools

SYSTEM_PROMPT = """You are a specialist MuleSoft researcher.

Your job is to gather accurate, up-to-date information on a given MuleSoft topic
using web search. You must:
- Search for official MuleSoft documentation, Salesforce Trailhead content, and
  authoritative community resources.
- Collect information on: core concepts, prerequisites, learning objectives,
  key APIs/connectors/tools involved, common pitfalls, and best practices.
- Return a structured research report with clear sections and cited sources.
- Focus only on information relevant to building a training course.

Always cite URLs. Prefer content from developer.mulesoft.com, trailhead.salesforce.com,
and docs.mulesoft.com.
"""

_USER_PROMPT_TEMPLATE = """\
Research the following MuleSoft topic for a training course:

**Topic:** {topic}

Provide a comprehensive research report covering:
1. Topic overview and relevance
2. Key concepts and terminology
3. Prerequisites learners need
4. Suggested learning objectives (measurable)
5. Core technical components (connectors, APIs, tools)
6. Common use cases and real-world examples
7. Common mistakes and how to avoid them
8. Recommended resources and documentation links
"""


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "web_search":
        return google_search.execute(
            query=tool_input["query"],
            num_results=tool_input.get("num_results", 5),
        )
    return f"Unknown tool: {tool_name}"


def run(topic: str, client: anthropic.Anthropic) -> str:
    """Run the researcher agent and return a structured research report."""
    thinking = thinking_param()
    output_config = output_config_param()

    create_kwargs: dict = dict(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        tools=[google_search.TOOL_DEFINITION],
    )
    if thinking:
        create_kwargs["thinking"] = thinking
    if output_config:
        create_kwargs["output_config"] = output_config

    messages = [{"role": "user", "content": _USER_PROMPT_TEMPLATE.format(topic=topic)}]

    return run_with_tools(
        client=client,
        create_kwargs=create_kwargs,
        messages=messages,
        tool_executor=_execute_tool,
        fallback="No research output generated.",
    )
