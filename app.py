"""
Caregiver Tutoring Assistant — Streamlit Frontend

Run with:
  streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHROMA_PERSIST_DIR, GEMINI_API_KEY, TRAIN_CSV
from data_loader import extract_misconception_docs, load_mathdial
from graph import build_pipeline, run_pipeline
from misconception_rag import MisconceptionRAG

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Caregiver Tutoring Assistant",
    page_icon="📚",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Caregiver script highlight box */
    .script-box {
        background: linear-gradient(135deg, #eef6ff 0%, #e0effe 100%);
        border-left: 5px solid #2563eb;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        font-size: 1.12rem;
        line-height: 1.75;
        color: #1e3a5f;
        margin: 6px 0 18px 0;
        font-style: italic;
    }

    /* Strategy badges */
    .badge {
        display: inline-block;
        padding: 4px 13px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-socratic         { background:#dbeafe; color:#1d4ed8; border:1px solid #bfdbfe; }
    .badge-hint             { background:#fef3c7; color:#92400e; border:1px solid #fde68a; }
    .badge-direct_correction{ background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }

    /* History entry */
    .history-item {
        border-left: 3px solid #e5e7eb;
        padding: 7px 12px;
        margin-bottom: 8px;
        font-size: 0.84rem;
        color: #6b7280;
    }

    /* Suppress default Streamlit padding on top */
    .block-container { padding-top: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline — cached so it loads only once per browser session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base (first run takes ~30 s)...")
def _get_pipeline():
    rag = MisconceptionRAG(persist_dir=CHROMA_PERSIST_DIR)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRAIN_CSV)
    if os.path.exists(csv_path):
        df = load_mathdial(csv_path)
        docs = extract_misconception_docs(df)
        rag.index_documents(docs)
    return build_pipeline(rag)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "phase": "setup",       # setup | attempt | result | done
    "problem": "",
    "ground_truth": "",
    "student_profile": "",
    "attempt_number": 1,
    "last_result": None,
    "history": [],          # list of dicts saved per attempt
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Strategy display metadata
# ---------------------------------------------------------------------------

_STRATEGY_META = {
    "socratic":          ("Socratic Question",    "badge-socratic",          "🔵"),
    "hint":              ("Hint",                 "badge-hint",              "🟡"),
    "direct_correction": ("Direct Correction",    "badge-direct_correction", "🔴"),
}

# ---------------------------------------------------------------------------
# Reusable: result card
# ---------------------------------------------------------------------------

def _result_card(result: dict, attempt: int):
    strategy    = result.get("strategy", "hint")
    label, badge_cls, dot = _STRATEGY_META.get(strategy, ("Hint", "badge-hint", "🟡"))
    response    = result.get("caregiver_response", "No response generated.")
    error       = result.get("error_state", {}) or {}
    frustration = result.get("frustration_level", "low")

    st.markdown(
        f'<span class="badge {badge_cls}">{dot} {label}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("**What to say to your student:**")
    st.markdown(f'<div class="script-box">{response}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Attempt #", attempt)
    c2.metric("Student mood", frustration.capitalize())
    c3.metric("Move type", label)

    with st.expander("Diagnostic details"):
        st.markdown(f"**Error type:** `{error.get('error_type', '-')}`")
        st.markdown(f"**Severity:** `{error.get('severity', '-')}`")
        st.markdown(f"**What went wrong:** {error.get('description', '-')}")
        st.markdown(f"**Core misconception:** {error.get('key_misconception', '-')}")
        st.markdown(f"**Detector confidence:** `{error.get('confidence', '-')}`")

# ---------------------------------------------------------------------------
# Reusable: sidebar history
# ---------------------------------------------------------------------------

def _sidebar_history():
    if not st.session_state.history:
        return
    with st.sidebar:
        st.header("Session history")
        for entry in st.session_state.history:
            r = entry["result"]
            strategy = r.get("strategy", "hint")
            label, badge_cls, dot = _STRATEGY_META.get(strategy, ("Hint", "badge-hint", "🟡"))
            with st.expander(f"Attempt {entry['attempt']} — {dot} {label}"):
                st.markdown(f"**Student wrote:**")
                st.caption(entry["student_attempt"][:300])
                st.markdown(f"**Script used:**")
                st.caption(r.get("caregiver_response", "-")[:300])

# ---------------------------------------------------------------------------
# Phase: SETUP
# ---------------------------------------------------------------------------

def _render_setup():
    st.title("📚 Caregiver Tutoring Assistant")
    st.caption(
        "Enter the problem below. The system will guide you on what to say "
        "to your student — without ever giving them the answer directly."
    )
    st.divider()

    with st.form("setup_form"):
        problem = st.text_area(
            "Math Problem",
            placeholder="Paste the full problem here…",
            height=130,
        )
        ground_truth = st.text_area(
            "Correct Solution / Answer",
            placeholder="Full step-by-step solution, or just the final answer…",
            height=90,
        )
        student_profile = st.text_input(
            "Student Profile  *(optional)*",
            placeholder="e.g. 7th grade, struggles with word problems",
        )

        if st.form_submit_button("Start session", type="primary", use_container_width=True):
            if not problem.strip():
                st.error("Please enter the math problem.")
            elif not ground_truth.strip():
                st.error("Please enter the correct solution.")
            else:
                st.session_state.problem        = problem.strip()
                st.session_state.ground_truth   = ground_truth.strip()
                st.session_state.student_profile = student_profile.strip() or "No profile available."
                st.session_state.attempt_number  = 1
                st.session_state.last_result     = None
                st.session_state.history         = []
                st.session_state.phase           = "attempt"
                st.rerun()

# ---------------------------------------------------------------------------
# Phase: ATTEMPT
# ---------------------------------------------------------------------------

def _render_attempt():
    col_title, col_badge = st.columns([4, 1])
    col_title.title("📚 Caregiver Tutoring Assistant")
    col_badge.metric("Attempt", st.session_state.attempt_number)

    with st.expander("Problem", expanded=True):
        st.markdown(st.session_state.problem)

    st.divider()

    with st.form("attempt_form"):
        student_attempt = st.text_area(
            "Student's answer / attempt",
            placeholder="Type exactly what the student wrote or said…",
            height=110,
        )
        caregiver_context = st.text_input(
            "How is the student feeling right now?  *(optional)*",
            placeholder="e.g. 'she seems frustrated', 'he has been crying', 'engaged and trying'",
        )

        if st.form_submit_button("Analyze", type="primary", use_container_width=True):
            if not student_attempt.strip():
                st.error("Please enter the student's attempt before analyzing.")
            else:
                pipeline = _get_pipeline()
                with st.spinner(f"Analyzing attempt {st.session_state.attempt_number}…"):
                    result = run_pipeline(
                        pipeline        = pipeline,
                        problem         = st.session_state.problem,
                        ground_truth    = st.session_state.ground_truth,
                        student_attempt = student_attempt.strip(),
                        student_profile = st.session_state.student_profile,
                        caregiver_context = caregiver_context.strip(),
                        attempt_number  = st.session_state.attempt_number,
                    )

                st.session_state.history.append({
                    "attempt":          st.session_state.attempt_number,
                    "student_attempt":  student_attempt.strip(),
                    "caregiver_context": caregiver_context.strip(),
                    "result":           result,
                })
                st.session_state.last_result = result
                st.session_state.phase       = "result"
                st.rerun()

    _sidebar_history()

# ---------------------------------------------------------------------------
# Phase: RESULT
# ---------------------------------------------------------------------------

def _render_result():
    col_title, col_badge = st.columns([4, 1])
    col_title.title("📚 Caregiver Tutoring Assistant")
    col_badge.metric("Attempt", st.session_state.attempt_number)

    with st.expander("Problem", expanded=False):
        st.markdown(st.session_state.problem)

    st.divider()
    _result_card(st.session_state.last_result, st.session_state.attempt_number)
    st.divider()

    st.markdown("**Did the student get it right after hearing the above?**")
    col_yes, col_no = st.columns(2)

    if col_yes.button("✅  Yes — student got it!", type="primary", use_container_width=True):
        st.session_state.phase = "done"
        st.rerun()

    if col_no.button("❌  No — still struggling", use_container_width=True):
        st.session_state.attempt_number += 1
        st.session_state.last_result     = None
        st.session_state.phase           = "attempt"
        st.rerun()

    _sidebar_history()

# ---------------------------------------------------------------------------
# Phase: DONE
# ---------------------------------------------------------------------------

def _render_done():
    st.title("📚 Caregiver Tutoring Assistant")
    st.divider()

    n = st.session_state.attempt_number
    st.success(
        f"Great work! The student solved it after **{n} attempt{'s' if n > 1 else ''}**. "
        "Keep up the good work!"
    )

    if st.session_state.history:
        st.subheader("Session summary")
        for entry in st.session_state.history:
            r        = entry["result"]
            strategy = r.get("strategy", "hint")
            label, badge_cls, dot = _STRATEGY_META.get(strategy, ("Hint", "badge-hint", "🟡"))
            snippet  = r.get("caregiver_response", "")[:140]
            st.markdown(
                f'<div class="history-item">'
                f'<strong>Attempt {entry["attempt"]}</strong> &nbsp;'
                f'<span class="badge {badge_cls}" style="font-size:0.7rem;padding:2px 8px">'
                f'{dot} {label}</span><br>'
                f'<em>"{snippet}…"</em>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    if st.button("Start a new problem", type="primary"):
        for k in list(_DEFAULTS.keys()):
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()

# ---------------------------------------------------------------------------
# Guard: API key
# ---------------------------------------------------------------------------

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it to a `.env` file in the project folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_phase = st.session_state.get("phase", "setup")
if   _phase == "attempt": _render_attempt()
elif _phase == "result":  _render_result()
elif _phase == "done":    _render_done()
else:                     _render_setup()
