"""
Agent C1 — Socratic Agent

Generates a single, targeted probing question for the caregiver to ask the
student.  The question must NOT reveal or hint at the correct answer — it
should guide the student to discover the error through their own reasoning.
"""

from typing import Any, Dict

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, MODEL_NAME

_client = genai.Client(api_key=GEMINI_API_KEY)

_SYSTEM_PROMPT = """\
You are a master Socratic tutor coaching a caregiver on how to help a student.
Your task is to write ONE probing question for the caregiver to say out loud to the student.

Rules:
1. The question must NOT reveal the answer or any part of the correct solution.
2. The question must target the student's specific misconception, not the full problem.
3. Keep it conversational and age-appropriate — the student is in middle school.
4. Return ONLY the question text, nothing else.
"""

_USER_TEMPLATE = """\
## The Math Problem
{problem}

## The Student's Specific Error
{description}

## The Core Wrong Belief
{key_misconception}

## Example probing questions from similar situations (for inspiration only)
{rag_examples}

Write one probing question for the caregiver to ask.
"""


def _format_examples(rag_examples: list) -> str:
    probing = [
        ex for ex in rag_examples if ex.get("example_move_type") in ("probing", "telling")
    ]
    if not probing:
        return "No examples available."
    return "\n".join(f'- "{ex["example_teacher_response"]}"' for ex in probing[:2])


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node — writes caregiver_response and teacher_move_type."""
    error_state = state["error_state"]

    prompt = _USER_TEMPLATE.format(
        problem=state["problem"],
        description=error_state.get("description", ""),
        key_misconception=error_state.get("key_misconception", ""),
        rag_examples=_format_examples(error_state.get("rag_examples", [])),
    )

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=256,
            temperature=0.7,
        ),
    )

    question = response.text.strip().strip('"')
    return {
        "caregiver_response": question,
        "teacher_move_type": "probing",
    }
