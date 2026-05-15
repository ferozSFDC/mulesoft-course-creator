"""Judge Agent — critiques research quality and completeness."""

from config import make_client, MODEL, thinking_param

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
- A final verdict: APPROVED (score ≥ 7), NEEDS_REVISION (score 5-6), or REJECTED (score < 5)
"""


def run(topic: str, research_report: str) -> str:
    """Evaluate a research report and return a structured critique."""
    client = make_client()
    thinking = thinking_param()

    create_kwargs = dict(
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

    response = client.messages.create(**create_kwargs)

    for block in response.content:
        if block.type == "text":
            return block.text

    return "No critique generated."


def extract_verdict(critique: str) -> str:
    """Parse verdict keyword from the judge's output."""
    for verdict in ("APPROVED", "NEEDS_REVISION", "REJECTED"):
        if verdict in critique.upper():
            return verdict
    return "NEEDS_REVISION"
