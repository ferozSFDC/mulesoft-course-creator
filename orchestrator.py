"""
Orchestrator Agent — manages the Course Creation System workflow.

Workflow:
  1. Researcher gathers topic information via web search.
  2. Judge critiques the research for quality and completeness.
  3. If REJECTED or NEEDS_REVISION, Researcher refines (up to MAX_RESEARCH_RETRIES).
  4. Content Builder produces the final structured course.
  5. Results are saved to the output/ directory.
"""

import json
import datetime
from pathlib import Path

from config import make_client, MODEL, thinking_param
from agents import researcher, judge, content_builder
from tools import google_search

MAX_RESEARCH_RETRIES = 2
OUTPUT_DIR = Path(__file__).parent / "output"

_REFINE_SYSTEM = (
    "You are a specialist MuleSoft researcher. You have received feedback on your "
    "previous research report and must produce an improved version that directly "
    "addresses every gap and weakness identified by the quality assessor. "
    "Use web search to fill any missing information."
)


def _save_artifact(name: str, content: str, run_id: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{run_id}_{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _print_step(step: str, detail: str = "") -> None:
    print(f"\n{'='*60}")
    print(f"  {step}")
    if detail:
        print(f"  {detail}")
    print(f"{'='*60}")


def _refine_research(topic: str, original_research: str, critique: str) -> str:
    """Ask the researcher to improve its report based on judge feedback."""
    client = make_client()
    thinking = thinking_param()

    messages = [
        {
            "role": "user",
            "content": (
                f"Topic: **{topic}**\n\n"
                "## Your Previous Research\n"
                f"{original_research}\n\n"
                "## Quality Assessor Feedback\n"
                f"{critique}\n\n"
                "Please revise and improve your research report, specifically addressing "
                "every weakness and gap the assessor identified. Use web search to fill "
                "any missing technical details."
            ),
        }
    ]

    create_kwargs = dict(
        model=MODEL,
        max_tokens=8192,
        system=_REFINE_SYSTEM,
        tools=[google_search.TOOL_DEFINITION],
        messages=messages,
    )
    if thinking:
        create_kwargs["thinking"] = thinking

    while True:
        response = client.messages.create(**create_kwargs)

        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if not tool_calls or response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return original_research

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


def run(topic: str) -> dict:
    """
    Orchestrate the full course creation pipeline.

    Returns a dict with keys: topic, research, critique, course, verdict, run_id.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n🎓 MuleSoft Course Creation System")
    print(f"   Topic  : {topic}")
    print(f"   Run ID : {run_id}")

    # ── Step 1: Research ──────────────────────────────────────────────────────
    _print_step("Step 1/3 — Researcher Agent", "Gathering information via web search…")
    research = researcher.run(topic)
    research_path = _save_artifact("research", research, run_id)
    print(f"  ✔ Research saved → {research_path.name}")

    # ── Step 2: Judge (with retry loop) ──────────────────────────────────────
    verdict = "NEEDS_REVISION"
    critique = ""
    attempt = 0

    while verdict != "APPROVED" and attempt <= MAX_RESEARCH_RETRIES:
        attempt += 1
        label = "Step 2/3" if attempt == 1 else f"Step 2/3 (retry {attempt-1})"
        _print_step(f"{label} — Judge Agent", "Evaluating research quality…")

        critique = judge.run(topic, research)
        verdict = judge.extract_verdict(critique)
        critique_path = _save_artifact(f"critique_attempt{attempt}", critique, run_id)
        print(f"  Verdict : {verdict}")
        print(f"  ✔ Critique saved → {critique_path.name}")

        if verdict in ("REJECTED", "NEEDS_REVISION") and attempt <= MAX_RESEARCH_RETRIES:
            _print_step(
                f"  ↻ Researcher refinement {attempt}/{MAX_RESEARCH_RETRIES}",
                "Incorporating judge feedback…",
            )
            research = _refine_research(topic, research, critique)
            research_path = _save_artifact(f"research_refined{attempt}", research, run_id)
            print(f"  ✔ Refined research saved → {research_path.name}")

    if verdict == "REJECTED":
        print(f"\n⚠️  Research could not reach approval after {MAX_RESEARCH_RETRIES} retries.")
        print("   Proceeding with best available research…")

    # ── Step 3: Content Builder ───────────────────────────────────────────────
    _print_step("Step 3/3 — Content Builder Agent", "Generating structured course…")
    course = content_builder.run(topic, research, critique)
    course_path = _save_artifact("course", course, run_id)
    print(f"  ✔ Course saved → {course_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    result = {
        "run_id": run_id,
        "topic": topic,
        "research": research,
        "critique": critique,
        "verdict": verdict,
        "course": course,
        "output_dir": str(OUTPUT_DIR),
    }

    summary_path = OUTPUT_DIR / f"{run_id}_summary.json"
    summary_path.write_text(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("research", "critique", "course")},
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_step("✅ Course Creation Complete")
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Final verdict    : {verdict}")
    print(f"  Course file      : {course_path.name}\n")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py \"<MuleSoft topic>\"")
        print("Example: python orchestrator.py \"MuleSoft DataWeave 2.0 Transformations\"")
        sys.exit(1)

    topic_input = " ".join(sys.argv[1:])
    run(topic_input)
