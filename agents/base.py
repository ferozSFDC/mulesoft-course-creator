"""Shared tool-loop logic for all agents that use Claude tool_use."""

from __future__ import annotations

import anthropic

MAX_TOOL_ITERATIONS = 10


def run_with_tools(
    client: anthropic.Anthropic,
    create_kwargs: dict,
    messages: list[dict],
    tool_executor: callable,
    fallback: str = "No output generated.",
) -> str:
    """
    Run the Claude tool-use loop until the model stops requesting tools or
    MAX_TOOL_ITERATIONS is reached.

    Args:
        client:         Configured Anthropic client.
        create_kwargs:  Base kwargs for client.messages.create (must not include 'messages').
        messages:       Initial message list (mutated internally; caller's list is not modified).
        tool_executor:  Callable(tool_name, tool_input) -> str that executes a tool call.
        fallback:       Returned if the model produces no text content.

    Returns:
        The final text block from the model's last response.
    """
    msgs = list(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(**create_kwargs, messages=msgs)

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if not tool_calls or response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return fallback

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": tool_executor(tc.name, tc.input),
            }
            for tc in tool_calls
        ]

        msgs = msgs + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]

    # Iteration cap hit — return whatever text the model last produced
    for block in response.content:
        if block.type == "text":
            return block.text
    return fallback
