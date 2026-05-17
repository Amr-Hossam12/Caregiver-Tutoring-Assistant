"""
Agent B — Orchestrator (Three-Condition Routing Logic)

Evaluates three strict conditions to decide the pedagogical strategy:

  Condition 1 — Error type
    Conceptual / procedural errors → candidate for Socratic (broken logic needs probing)
    Calculation errors             → always Hint or Direct Correction (no deep probing)

  Condition 2 — Frustration level  (parsed from caregiver_context)
    LOW / MEDIUM → Socratic questioning is appropriate
    HIGH         → Reduce cognitive load immediately (Hint or Direct Correction)

  Condition 3 — Attempt number  (graceful degradation)
    1–2  → Socratic (student still has mental energy, give them a chance)
    3    → Hint     (Socratic has not worked; give a targeted nudge)
    4+   → Direct Correction (stop the loop; explain it clearly)

Degradation ladder:  Socratic  →  Hint  →  Direct Correction
"""

import re
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Frustration detector (keyword / phrase heuristic — no LLM needed)
# ---------------------------------------------------------------------------

_HIGH_FRUSTRATION_SIGNALS = [
    "crying", "cries", "cry ", " cry",
    "give up", "giving up", "given up",
    "quit", "refuses", "won't try", "doesn't want to",
    "meltdown", "tantrum",
    "very frustrated", "extremely frustrated", "so frustrated",
    "been stuck", "stuck for an hour", "stuck for 20", "stuck for 30",
    "20 minutes", "30 minutes", "an hour", "hours",
    "can't do it", "cannot do it", "impossible",
    "hates", "hate this",
]

_MEDIUM_FRUSTRATION_SIGNALS = [
    "frustrated", "annoyed", "upset",
    "confused", "doesn't understand",
    "stuck for", "been trying",
    "multiple times", "several times", "keeps getting",
    "10 minutes", "15 minutes",
    "lost", "lost interest",
]


def _detect_frustration(caregiver_context: str) -> str:
    """Return 'high', 'medium', or 'low' based on caregiver's description."""
    if not caregiver_context:
        return "low"
    text = caregiver_context.lower()
    for phrase in _HIGH_FRUSTRATION_SIGNALS:
        if phrase in text:
            return "high"
    for phrase in _MEDIUM_FRUSTRATION_SIGNALS:
        if phrase in text:
            return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

_CONCEPTUAL_TYPES = {"conceptual_flaw", "procedural_error", "comprehension_error"}
_CALCULATION_TYPES = {"calculation_error"}


def _decide_strategy(
    error_type: str,
    frustration: str,
    attempt: int,
) -> str:
    """
    Apply the three conditions and return the strategy string.

    Graceful degradation:
      • Socratic     → requires ALL THREE conditions satisfied
      • Hint         → conceptual error that failed Socratic, or any calc error
      • Direct Corr  → attempt 4+, or high frustration on attempt 3+
    """
    is_conceptual = error_type in _CONCEPTUAL_TYPES

    # High frustration always reduces cognitive load
    if frustration == "high":
        return "direct_correction" if attempt >= 3 else "hint"

    # Calculation errors never get Socratic questioning
    if not is_conceptual:
        return "hint"

    # Conceptual/procedural: apply the three-condition gate
    if frustration in ("low", "medium") and attempt <= 2:
        return "socratic"            # All three conditions met
    elif attempt >= 4:
        return "direct_correction"   # Student is in a stuck loop
    else:
        return "hint"                # Attempt 3, or medium frustration


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrator node — evaluates conditions and writes 'strategy'."""
    error_state = state.get("error_state", {}) or {}
    error_type  = error_state.get("error_type", "conceptual_flaw")
    attempt     = int(state.get("attempt_number", 1))
    context     = state.get("caregiver_context", "")

    frustration = _detect_frustration(context)
    strategy    = _decide_strategy(error_type, frustration, attempt)

    print(
        f"[Orchestrator] error_type={error_type} | "
        f"frustration={frustration} | attempt={attempt} -> strategy={strategy}"
    )
    return {
        "strategy": strategy,
        "frustration_level": frustration,
    }
