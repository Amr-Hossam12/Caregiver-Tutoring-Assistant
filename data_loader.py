import pandas as pd
from typing import List, Dict, Any


def load_mathdial(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def parse_conversation(conversation: str) -> List[Dict[str, str]]:
    """Parse '|EOM|'-delimited MathDial conversation into structured turns."""
    turns = []
    for raw in conversation.split("|EOM|"):
        raw = raw.strip()
        if not raw:
            continue
        if raw.startswith("Teacher:"):
            content = raw[len("Teacher:"):].strip()
            move = "generic"
            if content.startswith("("):
                end = content.index(")")
                move = content[1:end].lower().strip()
                content = content[end + 1:].strip()
            turns.append({"role": "teacher", "move": move, "content": content})
        else:
            # Student turn — name varies per conversation
            parts = raw.split(":", 1)
            content = parts[1].strip() if len(parts) > 1 else raw
            turns.append({"role": "student", "move": "response", "content": content})
    return turns


def extract_misconception_docs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract one document per row for RAG indexing.
    Each document captures the misconception label, student attempt, and a
    representative teacher move from the conversation.
    """
    docs = []
    for _, row in df.iterrows():
        confusion = row.get("teacher_described_confusion", "")
        if pd.isna(confusion) or not str(confusion).strip():
            continue

        turns = parse_conversation(str(row["conversation"]))
        # Prefer a probing or telling move as the canonical example response
        example_turn = next(
            (t for t in turns if t["role"] == "teacher" and t["move"] in ("probing", "telling", "hint")),
            next((t for t in turns if t["role"] == "teacher"), None),
        )
        example_response = example_turn["content"] if example_turn else ""
        example_move_type = example_turn["move"] if example_turn else "generic"

        docs.append({
            "qid": str(row["qid"]),
            "misconception": str(confusion).strip(),
            "student_attempt": str(row["student_incorrect_solution"]),
            "problem": str(row["question"]),
            "ground_truth": str(row["ground_truth"]),
            "student_profile": str(row.get("student_profile", "")),
            "example_teacher_response": example_response,
            "example_move_type": example_move_type,
            "self_correctness": str(row.get("self-correctness", "")),
        })
    return docs
