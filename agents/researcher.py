"""Researcher Agent — finds up-to-date MuleSoft information via web search."""

from config import make_client, MODEL, thinking_param
from tools import google_search

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


def run(topic: str) -> str:
    """Run the researcher agent and return a structured research report."""
    client = make_client()
    thinking = thinking_param()

    messages = [
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(topic=topic)}
    ]

    create_kwargs = dict(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        tools=[google_search.TOOL_DEFINITION],
        messages=messages,
    )
    if thinking:
        create_kwargs["thinking"] = thinking

    while True:
        response = client.messages.create(**create_kwargs)

        # Collect any tool calls from this response
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if not tool_calls or response.stop_reason == "end_turn":
            # No more tool calls — extract final text
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No research output generated."

        # Execute each tool call and build tool_result blocks
        assistant_msg = {"role": "assistant", "content": response.content}
        tool_results = []
        for tool_call in tool_calls:
            args = tool_call.input
            result_text = google_search.execute(
                query=args["query"],
                num_results=args.get("num_results", 5),
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_text,
                }
            )

        messages = messages + [
            assistant_msg,
            {"role": "user", "content": tool_results},
        ]
        create_kwargs["messages"] = messages
