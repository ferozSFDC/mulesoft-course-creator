"""Content Builder Agent — turns approved research into a structured course."""

from config import make_client, MODEL, thinking_param

SYSTEM_PROMPT = """You are an expert MuleSoft instructional designer.

Your job is to transform research reports into fully structured, production-ready
MuleSoft training courses. You create engaging, pedagogically sound content that
follows adult learning principles.

Course structure you must produce:
1. **Course Overview** — title, description, target audience, duration estimate
2. **Prerequisites** — what learners must know before starting
3. **Learning Objectives** — 5-8 SMART objectives
4. **Module Breakdown** — 4-6 modules, each with:
   - Module title and description
   - Subtopics covered
   - Hands-on lab/exercise description
   - Knowledge check questions (2-3 per module)
5. **Capstone Project** — a real-world integration scenario tying all modules together
6. **Assessment Strategy** — how learners demonstrate mastery
7. **Additional Resources** — curated links, docs, community forums
8. **Instructor Notes** — tips for delivering this course effectively

Make the content practical, Salesforce/MuleSoft certification-aligned where applicable,
and suitable for professional developers.
"""


def run(topic: str, research_report: str, critique: str) -> str:
    """Build a complete course structure from research and judge feedback."""
    client = make_client()
    thinking = thinking_param()

    create_kwargs = dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Build a complete MuleSoft training course on: **{topic}**\n\n"
                    "## Research Report\n"
                    f"{research_report}\n\n"
                    "## Quality Assessment & Recommendations\n"
                    f"{critique}\n\n"
                    "Using the research and addressing the judge's recommendations, "
                    "produce a full course structure following your instructional design "
                    "framework. The course should be comprehensive, practical, and "
                    "immediately usable by a MuleSoft training team."
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

    return "No course content generated."
