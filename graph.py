"""
LangGraph state machine wiring all agents together.

Flow:
  misconception_detector
      -> orchestrator
          -> socratic_agent       (conceptual error, low/medium frustration, attempt 1-2)
          -> hint_agent           (calc error, attempt 3, or medium frustration)
          -> direct_correction_agent  (attempt 4+, or high frustration)
"""

from typing import Any, Dict, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from misconception_rag import MisconceptionRAG
import agents.misconception_detector as detector_agent
import agents.orchestrator as orchestrator_agent
import agents.socratic_agent as socratic_agent
import agents.hint_agent as hint_agent
import agents.direct_correction_agent as correction_agent


# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    problem: str
    ground_truth: str
    student_attempt: str
    student_profile: str

    # NEW: caregiver's free-text note about the session (frustration signals, etc.)
    caregiver_context: Optional[str]

    # NEW: which attempt number this is (1 = first try; drives graceful degradation)
    attempt_number: Optional[int]

    # ── Agent A output ───────────────────────────────────────────────────────
    error_state: Optional[Dict[str, Any]]

    # ── Agent B output ───────────────────────────────────────────────────────
    strategy: Optional[str]           # "socratic" | "hint" | "direct_correction"
    frustration_level: Optional[str]  # "low" | "medium" | "high"

    # ── Agent C output ───────────────────────────────────────────────────────
    caregiver_response: Optional[str]
    teacher_move_type: Optional[str]  # "probing" | "hint" | "direct_correction"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_detector_node(rag: MisconceptionRAG):
    def node(state: PipelineState) -> Dict[str, Any]:
        print("[Pipeline] Running Misconception Detector ...")
        return detector_agent.run(state, rag)
    return node


def _orchestrator_node(state: PipelineState) -> Dict[str, Any]:
    print("[Pipeline] Running Orchestrator ...")
    return orchestrator_agent.run(state)


def _socratic_node(state: PipelineState) -> Dict[str, Any]:
    print("[Pipeline] Running Socratic Agent ...")
    return socratic_agent.run(state)


def _hint_node(state: PipelineState) -> Dict[str, Any]:
    print("[Pipeline] Running Hint Agent ...")
    return hint_agent.run(state)


def _correction_node(state: PipelineState) -> Dict[str, Any]:
    print("[Pipeline] Running Direct Correction Agent ...")
    return correction_agent.run(state)


# ---------------------------------------------------------------------------
# Routing condition
# ---------------------------------------------------------------------------

def _route(
    state: PipelineState,
) -> Literal["socratic_agent", "hint_agent", "direct_correction_agent"]:
    strategy = state.get("strategy", "hint")
    if strategy == "socratic":
        return "socratic_agent"
    elif strategy == "direct_correction":
        return "direct_correction_agent"
    else:
        return "hint_agent"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline(rag: MisconceptionRAG) -> Any:
    graph = StateGraph(PipelineState)

    graph.add_node("misconception_detector", _make_detector_node(rag))
    graph.add_node("orchestrator", _orchestrator_node)
    graph.add_node("socratic_agent", _socratic_node)
    graph.add_node("hint_agent", _hint_node)
    graph.add_node("direct_correction_agent", _correction_node)

    graph.set_entry_point("misconception_detector")
    graph.add_edge("misconception_detector", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _route,
        {
            "socratic_agent": "socratic_agent",
            "hint_agent": "hint_agent",
            "direct_correction_agent": "direct_correction_agent",
        },
    )
    graph.add_edge("socratic_agent", END)
    graph.add_edge("hint_agent", END)
    graph.add_edge("direct_correction_agent", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    pipeline,
    problem: str,
    ground_truth: str,
    student_attempt: str,
    student_profile: str = "No profile available.",
    caregiver_context: str = "",
    attempt_number: int = 1,
) -> PipelineState:
    initial_state: PipelineState = {
        "problem": problem,
        "ground_truth": ground_truth,
        "student_attempt": student_attempt,
        "student_profile": student_profile,
        "caregiver_context": caregiver_context,
        "attempt_number": attempt_number,
        "error_state": None,
        "strategy": None,
        "frustration_level": None,
        "caregiver_response": None,
        "teacher_move_type": None,
    }
    return pipeline.invoke(initial_state)
