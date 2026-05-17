# Session Handoff

## Goal

Build a production-ready **Multi-Agent Caregiver Tutoring Assistant** that sits between a caregiver and a student during math homework sessions.

The system must:
- Accept a math problem, the correct solution, and the student's incorrect attempt
- Diagnose the student's specific cognitive error using RAG over the MathDial dataset
- Route to one of three pedagogical strategies based on **error type + frustration level + attempt number**
- Output a ready-to-read caregiver script — a question, hint, or correction — that never reveals the answer
- Track attempt count automatically and degrade gracefully: Socratic → Hint → Direct Correction

---

## Current State of the Code

### What is fully working
- **LangGraph pipeline** (`graph.py`) — 4-node state machine, tested end-to-end via CLI
- **ChromaDB RAG** (`misconception_rag.py`) — 2253 MathDial docs indexed, persisted in `chroma_db/`
- **Misconception Detector** (`agents/misconception_detector.py`) — RAG + Gemini → structured JSON error state
- **Orchestrator** (`agents/orchestrator.py`) — 3-condition routing with keyword frustration detection
- **Socratic Agent** (`agents/socratic_agent.py`) — generates probing questions
- **Hint Agent** (`agents/hint_agent.py`) — generates one-step hints
- **Direct Correction Agent** (`agents/direct_correction_agent.py`) — explicit correction for stuck students
- **CLI** (`main.py`) — `--demo`, `--attempt N`, `--frustration`, `--context` flags all work
- **GitHub repo** — pushed and organised at `https://github.com/Amr-Hossam12/Caregiver-Tutoring-Assistant`

### What is NOT working
- **Streamlit frontend** (`app.py`) — user reported a blank page when running `python -m streamlit run app.py`. Root cause not yet confirmed; session was interrupted before investigation.

---

## Active Files (Last Edited This Session)

| File | Last change |
|---|---|
| `app.py` | Streamlit frontend — **blank page bug unresolved** |
| `agents/orchestrator.py` | Full 3-condition rewrite with frustration detector |
| `agents/direct_correction_agent.py` | New file — 3rd pedagogical tier |
| `graph.py` | Added `caregiver_context`, `attempt_number`, `frustration_level` to state; added `direct_correction_agent` node |
| `main.py` | Added `--attempt`, `--context`, `--frustration` CLI flags |
| `config.py` | Updated `TRAIN_CSV`/`TEST_CSV` to `data/train.csv` / `data/test.csv` |
| `README.md` | Unified doc covering both phases |

---

## Everything That Failed

### 1. Gemini model quota (`gemini-2.0-flash`)
The `.env.example` key (`AIzaSyB9...`) hit its **free-tier daily quota** on first run.  
**Fix applied:** switched `MODEL_NAME` in `config.py` to `gemini-flash-lite-latest`, which had available quota.  
**Current working key** is in `.env` (not committed). The `.env.example` still has the old exhausted key — it was intentionally committed because the user asked for it.

### 2. Unicode crash on Windows console
`agents/orchestrator.py` used `→` (U+2192) and `main.py` used `─` (U+2500) in print statements.  
Windows CP1252 console threw `UnicodeEncodeError`.  
**Fix applied:** replaced all Unicode symbols with plain ASCII (`->`, `-`).

### 3. ChromaDB ONNX model download (first run)
First run triggered a ~80 MB download of `all-MiniLM-L6-v2` ONNX model.  
The Bash tool timed out mid-download.  
**Not a bug** — subsequent runs use the cached model from `C:\Users\Amr\.cache\chroma\`.

### 4. `python main.py` appeared to freeze
The `multiline_input()` function waits for a blank line to end input.  
User did not know they needed to press **Enter on an empty line** to proceed.  
**Partial fix:** updated prompt to show `(type your text, then press Enter on an EMPTY line to continue)` and added a `> ` prefix. The user interrupted before this edit was accepted.

### 5. Streamlit `streamlit` command not on PATH
`streamlit run app.py` failed with "not recognized".  
**Fix:** use `python -m streamlit run app.py` instead.

### 6. `.env.example` file name typo
User typed `app.p` instead of `app.py` in the streamlit command, causing "not a .py file" error.

### 7. Git push rejected (remote had existing commits)
Remote repo already had a `requirements.txt` from Phase 1.  
**Fix:** `git pull --rebase`, resolved the `requirements.txt` conflict manually, then pushed.

### 8. `.env.example` deleted by a remote commit
Between two push sessions, a remote commit deleted `.env.example`.  
**Fix:** recreated and re-committed it.

---

## Next Step to Take

**Investigate and fix the blank Streamlit page.**

Most likely causes (check in this order):

1. **Session state init runs before the page renders** — the `for k, v in _DEFAULTS.items()` loop at module level may execute before Streamlit's context is ready. Move all state initialisation inside a `def _init_state()` function called at the top of each phase renderer, or use `st.session_state.setdefault()`.

2. **The router dict at the bottom fails silently** — the final line:
   ```python
   {
       "setup": _render_setup,
       ...
   }.get(st.session_state.phase, _render_setup)()
   ```
   If `st.session_state` is not accessible at module level (e.g., on cold start), `.phase` throws a `KeyError` before any UI renders. Replace with an explicit `if/elif` block:
   ```python
   phase = st.session_state.get("phase", "setup")
   if phase == "setup":
       _render_setup()
   elif phase == "attempt":
       _render_attempt()
   ...
   ```

3. **`@st.cache_resource` call at import time** — `_get_pipeline()` is decorated but not called at import; should be fine. If it was accidentally called at module level it would block rendering.

**Recommended fix:**
```python
# Replace the module-level router dict with this at the bottom of app.py:
phase = st.session_state.get("phase", "setup")
if   phase == "setup":   _render_setup()
elif phase == "attempt": _render_attempt()
elif phase == "result":  _render_result()
elif phase == "done":    _render_done()
else:                    _render_setup()
```

---

## Key Constants to Know

| Thing | Value |
|---|---|
| Working Gemini model | `gemini-flash-lite-latest` |
| Working API key | In `.env` (not committed) |
| Exhausted key (in `.env.example`) | `AIzaSyB9_CMsHGS8NQdwXaUidA_-dYxmJxKn3tE` |
| ChromaDB collection | `mathdial_misconceptions` (2253 docs) |
| Dataset location | `data/train.csv`, `data/test.csv` |
| GitHub repo | `https://github.com/Amr-Hossam12/Caregiver-Tutoring-Assistant` |
