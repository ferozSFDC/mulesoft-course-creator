"""Judge Agent — critiques research quality and completeness."""

from __future__ import annotations

import re

import anthropic

from config import MODEL, thinking_param, output_config_param

SYSTEM_PROMPT = """You are a senior MuleSoft curriculum quality assessor.

Your job is to critically evaluate research reports intended for MuleSoft training
courses. You must assess:

1. **Accuracy** — Is the technical information correct and current?
2. **Completeness** — Are all essential concepts covered? What is missing?
3. **Pedagogical suitability** — Is the content appropriate for a training course?
4. **Learning objective quality** — Are objectives specific, measurable, achievable?
5. **Source reliability** — Are cited sources authoritative and trustworthy?
6. **Scope appropriateness** — Is the topic scoped correctly (not too broad/narrow)?
7. **Practical applicability** — Does it include hands-on, real-world relevance?

Provide a structured critique with:
- An overall quality score (1-10) with justification
- Specific strengths (at least 3)
- Specific gaps or weaknesses (at least 3)
- Concrete recommendations for improvement
- A final verdict on its own line in this exact format:
  VERDICT: APPROVED
  or
  VERDICT: NEEDS_REVISION
  or
  VERDICT: REJECTED

Use APPROVED for score >= 7, NEEDS_REVISION for score 5-6, REJECTED for score < 5.
"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVED|NEEDS_REVISION|REJECTED)", re.IGNORECASE)


def run(topic: str, research_report: str, client: anthropic.Anthropic) -> str:
    """Evaluate a research report and return a structured critique."""
    thinking = thinking_param()
    output_config = output_config_param()

    create_kwargs: dict = dict(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Please evaluate this research report for a MuleSoft training course "
                    f"on the topic: **{topic}**\n\n"
                    "---\n"
                    f"{research_report}\n"
                    "---\n\n"
                    "Provide your critique following the assessment criteria in your instructions."
                ),
            }
        ],
    )
    if thinking:
        create_kwargs["thinking"] = thinking
    if output_config:
        create_kwargs["output_config"] = output_config

    response = client.messages.create(**create_kwargs)

    for block in response.content:
        if block.type == "text":
            return block.text

    return "No critique generated."


def extract_verdict(critique: str) -> str:
    """
    Parse the verdict from the judge's structured output.

    Looks for the canonical 'VERDICT: <value>' line produced by the system
    prompt. Falls back to NEEDS_REVISION if not found.
    """
    match = _VERDICT_RE.search(critique)
    if match:
        return match.group(1).upper()
    return "NEEDS_REVISION"
