"""
Agent C2 — Hint Agent

Generates a small, targeted clue for the caregiver to share with the student.
The hint advances the student exactly ONE step without revealing the final answer.
"""

from typing import Any, Dict

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MODEL_NAME

_client = genai.Client(api_key=GEMINI_API_KEY)

_SYSTEM_PROMPT = """\
You are a tutoring assistant helping a caregiver support a struggling student.
Your task is to write ONE short hint that the caregiver can share with the student.

Rules:
1. The hint must move the student forward by exactly ONE step, not solve the problem.
2. Do NOT state the final answer or the full solution.
3. Reference only the next immediate sub-step the student needs to fix.
4. Keep it encouraging and simple — the student is in middle school.
5. Return ONLY the hint text, nothing else.
"""

_USER_TEMPLATE = """\
## The Math Problem
{problem}

## The Student's Incorrect Attempt
{student_attempt}

## The Student's Specific Error
{description}

## First step of the correct solution (use only to craft the hint — do NOT repeat verbatim)
{first_correct_step}

Write one short hint for the caregiver to give the student.
"""


def _extract_first_step(ground_truth: str) -> str:
    lines = [l.strip() for l in ground_truth.splitlines() if l.strip()]
    return lines[0] if lines else ground_truth[:200]


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node — writes caregiver_response and teacher_move_type."""
    error_state = state["error_state"]

    prompt = _USER_TEMPLATE.format(
        problem=state["problem"],
        student_attempt=state["student_attempt"],
        description=error_state.get("description", ""),
        first_correct_step=_extract_first_step(state.get("ground_truth", "")),
    )

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=256,
            temperature=0.5,
        ),
    )

    hint = response.text.strip().strip('"')
    return {
        "caregiver_response": hint,
        "teacher_move_type": "hint",
    }
