"""
Agent A — Misconception Detector (The Diagnostician)

Uses RAG to surface similar historical errors, then asks Gemini to classify
the student's specific cognitive gap and return a structured Error State JSON.
"""

import json
import re
from typing import Any, Dict

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MODEL_NAME
from misconception_rag import MisconceptionRAG

_client = genai.Client(api_key=GEMINI_API_KEY)

_SYSTEM_PROMPT = """\
You are a pedagogical diagnostic AI embedded in a tutoring assistant.
Your job is to analyse a student's incorrect solution and identify the precise cognitive error.
You must return ONLY a valid JSON object — no markdown, no prose around it.
"""

_USER_TEMPLATE = """\
## Math Problem
{problem}

## Correct Solution
{ground_truth}

## Student's Incorrect Attempt
{student_attempt}

## Student Profile
{student_profile}

## Similar Misconceptions Retrieved from Database
{rag_examples}

---
Classify the student's error and return a JSON object with exactly these keys:
{{
  "error_type": "<one of: conceptual_flaw | calculation_error | procedural_error | comprehension_error>",
  "description": "<one sentence describing exactly what the student did wrong>",
  "key_misconception": "<the core wrong belief or misapplied rule>",
  "confidence": <float 0.0–1.0>,
  "severity": "<minor | moderate | major>"
}}
"""


def _format_rag_examples(examples: list) -> str:
    if not examples:
        return "No similar examples found."
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"{i}. Misconception: {ex['misconception']}\n"
            f"   Teacher response ({ex['example_move_type']}): {ex['example_teacher_response']}\n"
            f"   Student self-corrected: {ex['self_correctness']}"
        )
    return "\n".join(lines)


def run(state: Dict[str, Any], rag: MisconceptionRAG) -> Dict[str, Any]:
    """LangGraph node — detects the misconception and writes error_state."""
    rag_hits = rag.query(
        student_attempt=state["student_attempt"],
        problem=state["problem"],
        n_results=3,
    )

    prompt = _USER_TEMPLATE.format(
        problem=state["problem"],
        ground_truth=state["ground_truth"],
        student_attempt=state["student_attempt"],
        student_profile=state.get("student_profile", "No profile available."),
        rag_examples=_format_rag_examples(rag_hits),
    )

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=512,
            temperature=0.2,
        ),
    )

    raw = response.text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        error_state = json.loads(raw)
    except json.JSONDecodeError:
        error_state = {
            "error_type": "conceptual_flaw",
            "description": raw[:300],
            "key_misconception": "Unable to parse structured output.",
            "confidence": 0.5,
            "severity": "moderate",
        }

    error_state["rag_examples"] = rag_hits
    return {"error_state": error_state}
