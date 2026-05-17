"""
Caregiver Tutoring Assistant -- CLI Entry Point

Usage:
  python main.py                                 # interactive mode
  python main.py --demo                          # first sample from test.csv (attempt 1, no context)
  python main.py --demo --index N                # Nth sample (0-based)
  python main.py --demo --index N --attempt A    # simulate attempt number A
  python main.py --demo --index N --frustration  # simulate high-frustration session
  python main.py --reindex                       # force re-embed the MathDial training data
"""

import argparse
import os
import sys
import textwrap

from config import GEMINI_API_KEY, CHROMA_PERSIST_DIR, TEST_CSV, TRAIN_CSV
from data_loader import extract_misconception_docs, load_mathdial
from graph import build_pipeline, run_pipeline
from misconception_rag import MisconceptionRAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key():
    if not GEMINI_API_KEY:
        print(
            "[ERROR] GEMINI_API_KEY is not set.\n"
            "  Create a .env file with: GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)


def _print_banner():
    print("\n" + "=" * 60)
    print("  Caregiver Tutoring Assistant  |  Multi-Agent Pipeline")
    print("=" * 60 + "\n")


def _print_result(result: dict):
    SEP = "-" * 60
    error = result.get("error_state", {}) or {}

    print("\n" + SEP)
    print("DIAGNOSTIC REPORT")
    print(SEP)
    print(f"  Error type   : {error.get('error_type', '-')}")
    print(f"  Severity     : {error.get('severity', '-')}")
    print(f"  Description  : {error.get('description', '-')}")
    print(f"  Misconception: {error.get('key_misconception', '-')}")
    print(f"  Confidence   : {error.get('confidence', '-')}")

    print()
    print("ROUTING DECISION")
    print(SEP)
    strategy    = result.get("strategy", "-")
    frustration = result.get("frustration_level", "-")
    attempt     = result.get("attempt_number", "-")
    move        = result.get("teacher_move_type", "-")
    print(f"  Strategy     : {strategy.upper() if strategy else '-'}")
    print(f"  Frustration  : {frustration}")
    print(f"  Attempt      : {attempt}")
    print(f"  Move type    : {move}")

    print()
    print("CAREGIVER SCRIPT")
    print(SEP)
    response = result.get("caregiver_response", "No response generated.")
    print(textwrap.fill(response, width=56, initial_indent="  ", subsequent_indent="  "))
    print(SEP + "\n")


def _initialise_rag(force_reindex: bool = False) -> MisconceptionRAG:
    rag = MisconceptionRAG(persist_dir=CHROMA_PERSIST_DIR)
    csv_path = os.path.join(os.path.dirname(__file__), TRAIN_CSV)
    if not os.path.exists(csv_path):
        print(f"[WARNING] {TRAIN_CSV} not found -- RAG will be empty.")
        return rag
    print("[Setup] Loading MathDial training data ...")
    df = load_mathdial(csv_path)
    docs = extract_misconception_docs(df)
    rag.index_documents(docs, force=force_reindex)
    return rag


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def run_demo(
    pipeline,
    idx: int = 0,
    attempt_number: int = 1,
    caregiver_context: str = "",
):
    csv_path = os.path.join(os.path.dirname(__file__), TEST_CSV)
    if not os.path.exists(csv_path):
        print(f"[ERROR] {TEST_CSV} not found.")
        return

    df = load_mathdial(csv_path)
    if idx >= len(df):
        print(f"[ERROR] Index {idx} out of range (test set has {len(df)} rows).")
        return

    row = df.iloc[idx]
    print(f"\n[Demo] test sample index={idx}, qid={row['qid']}, "
          f"attempt={attempt_number}, context='{caregiver_context or 'none'}'\n")

    print("PROBLEM:")
    print(textwrap.fill(str(row["question"]), width=60))
    print("\nSTUDENT ATTEMPT:")
    print(textwrap.fill(str(row["student_incorrect_solution"]), width=60))

    result = run_pipeline(
        pipeline=pipeline,
        problem=str(row["question"]),
        ground_truth=str(row["ground_truth"]),
        student_attempt=str(row["student_incorrect_solution"]),
        student_profile=str(row.get("student_profile", "")),
        caregiver_context=caregiver_context,
        attempt_number=attempt_number,
    )

    # Stitch attempt_number back in for display (it's an input, not written by agents)
    result["attempt_number"] = attempt_number
    _print_result(result)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(pipeline):
    print("Enter the details below (blank line to finish multi-line fields).\n")

    def multiline_input(label: str) -> str:
        print(f"{label}:")
        print("  (type your text, then press Enter on an EMPTY line to continue)")
        lines = []
        while True:
            line = input("  > ")
            if line == "":
                if lines:          # only stop once at least one line has been entered
                    break
                print("  (please enter some text first)")
            else:
                lines.append(line)
        return "\n".join(lines).strip()

    problem        = multiline_input("Math Problem")
    ground_truth   = multiline_input("Correct Solution / Answer")
    student_attempt = multiline_input("Student's Incorrect Attempt")

    student_profile = input(
        "Student Profile (optional, press Enter to skip): "
    ).strip() or "No profile available."

    caregiver_context = input(
        "Any notes about the student right now? "
        "(e.g. 'she is frustrated', 'he has been crying') [optional]: "
    ).strip()

    attempt_str = input("Which attempt is this? (1, 2, 3 ... default=1): ").strip()
    attempt_number = int(attempt_str) if attempt_str.isdigit() else 1

    result = run_pipeline(
        pipeline=pipeline,
        problem=problem,
        ground_truth=ground_truth,
        student_attempt=student_attempt,
        student_profile=student_profile,
        caregiver_context=caregiver_context,
        attempt_number=attempt_number,
    )
    result["attempt_number"] = attempt_number
    _print_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Caregiver Tutoring Assistant MAS")
    parser.add_argument("--demo",        action="store_true", help="Run a sample from test.csv")
    parser.add_argument("--index",       type=int, default=0, help="Row index in test.csv (default: 0)")
    parser.add_argument("--attempt",     type=int, default=1, help="Attempt number (default: 1)")
    parser.add_argument("--context",     type=str, default="",
                        help="Caregiver context string (e.g. 'she is crying')")
    parser.add_argument("--frustration", action="store_true",
                        help="Shortcut: set context to 'student has been crying for 20 minutes'")
    parser.add_argument("--reindex",     action="store_true", help="Force re-embed training data")
    args = parser.parse_args()

    _check_api_key()
    _print_banner()

    rag = _initialise_rag(force_reindex=args.reindex)
    pipeline = build_pipeline(rag)

    caregiver_context = args.context
    if args.frustration:
        caregiver_context = "student has been crying for 20 minutes and wants to give up"

    if args.demo:
        run_demo(
            pipeline,
            idx=args.index,
            attempt_number=args.attempt,
            caregiver_context=caregiver_context,
        )
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
