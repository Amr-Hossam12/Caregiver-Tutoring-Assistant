"""
Agent C3 — Direct Correction Agent

Activated when the pedagogical ladder has been exhausted:
  • Attempt 4+ (student has not responded to Socratic or Hints), OR
  • High frustration on attempt 3+ (student is overwhelmed)

Provides an explicit, kind correction that tells the student exactly what
went wrong and gives the precise next step — without revealing the full answer.
"""

from typing import Any, Dict

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MODEL_NAME

_client = genai.Client(api_key=GEMINI_API_KEY)

_SYSTEM_PROMPT = """\
You are a compassionate tutor helping a student who has been genuinely stuck for a while.
The caregiver needs a clear, direct correction to share — not another hint or question.

Rules:
1. In ONE sentence, name the specific error the student made (be clear, not harsh).
2. In ONE sentence, give the exact correct next step they should take.
3. End with one brief encouraging remark (one short sentence).
4. Do NOT solve the full problem.
5. Return ONLY these three sentences, nothing else.
"""

_USER_TEMPLATE = """\
## The Math Problem
{problem}

## The Student's Incorrect Attempt
{student_attempt}

## The Specific Error
{description}

## The Core Wrong Belief
{key_misconception}

## First correct step (reference only — do NOT copy verbatim)
{first_correct_step}

Write a direct, kind correction for the caregiver to read to the student.
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
        key_misconception=error_state.get("key_misconception", ""),
        first_correct_step=_extract_first_step(state.get("ground_truth", "")),
    )

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=256,
            temperature=0.4,
        ),
    )

    correction = response.text.strip().strip('"')
    return {
        "caregiver_response": correction,
        "teacher_move_type": "direct_correction",
    }
