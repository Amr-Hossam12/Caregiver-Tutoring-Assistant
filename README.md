# Educational Math Hint Generation: Evaluating Agentic Architectures

**Course:** NLP Applications — Nile University  
**Authors:** Hossam Nasr & Mohamed Nashaat Ibrahim

---

## Project Abstract

When caregivers use LLMs to help children with math homework, standard Zero-Shot prompting frequently results in **"Answer Leakage"** — giving away the final answer rather than providing Socratic scaffolding.

This project tackles this problem in two phases:

- **Phase 1:** Benchmark three advanced NLP architectures (Self-Refinement, Multi-Agent Validation, RAG) using the Groq API.
- **Phase 2:** Build a production-ready **Multi-Agent System (MAS)** caregiver assistant grounded in the MathDial dataset, with a full Streamlit frontend.

---

## Repository Structure

```
├── .env.example                    # API key template — copy to .env
├── requirements.txt                # All dependencies
│
├── app.py                          # Phase 2: Streamlit web frontend
├── main.py                         # Phase 2: CLI entry point
├── config.py                       # Phase 2: Configuration
├── data_loader.py                  # Phase 2: MathDial data parser
├── graph.py                        # Phase 2: LangGraph pipeline
├── misconception_rag.py            # Phase 2: ChromaDB RAG store
│
├── agents/                         # Phase 2: MAS agents
│   ├── misconception_detector.py   #   Agent A — RAG + LLM diagnosis
│   ├── orchestrator.py             #   Agent B — 3-condition routing
│   ├── socratic_agent.py           #   Agent C1 — probing questions
│   ├── hint_agent.py               #   Agent C2 — targeted hints
│   └── direct_correction_agent.py  #   Agent C3 — explicit correction
│
├── data/                           # MathDial dataset (Phase 2)
│   ├── train.csv
│   └── test.csv
│
├── phase1/                         # Phase 1 experiment scripts
│   ├── approach1_self_refine.py
│   ├── approach2_validation_loop.py
│   ├── approach3_rag_agent.py
│   ├── Judge LLM.py
│   └── NLP_Apps_phase1.ipynb
│
└── results/                        # Phase 1 benchmark results
    ├── Approach1_SelfRefine_Results.xlsx
    ├── Approach2_ValidationLoop_Results.xlsx
    └── Approach3_ValidationLoop(KB)_Results.xlsx
```

---

## Phase 2 — MAS Caregiver Tutoring Assistant

### Setup

```bash
pip install -r requirements.txt
```

Copy the API key template and add your Gemini key:

```bash
cp .env.example .env
```

The `.env.example` file contains a working Gemini API key.

### Run (Web UI)

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. Enter the math problem, correct solution, and the student's attempt. The system tracks attempt count automatically and asks if the student got it right after each response.

### Run (CLI)

```bash
python main.py --demo --index 0        # first test sample
python main.py --demo --index N        # Nth sample (0-based)
python main.py --demo --attempt 3      # simulate attempt 3
python main.py --demo --frustration    # simulate high-frustration session
python main.py                         # interactive mode
```

### Pipeline Architecture

```
Student Attempt
      │
      ▼
Misconception Detector  ←── ChromaDB RAG (2253 MathDial examples)
      │
      ▼  error_state: {type, severity, description, misconception}
Orchestrator (3-Condition Router)
      │
      ├── Condition 1: error type (conceptual vs. calculation)
      ├── Condition 2: frustration level (from caregiver note)
      └── Condition 3: attempt number (graceful degradation)
      │
      ├──► Socratic Agent     (attempt 1–2, conceptual, low frustration)
      ├──► Hint Agent         (attempt 3, calc error, or medium frustration)
      └──► Direct Correction  (attempt 4+, or high frustration)
      │
      ▼
Caregiver Script  →  displayed to caregiver
```

---

## Phase 1 — Benchmark Results

| Approach | Student Pass Rate | API Calls | Verdict |
|---|---|---|---|
| 1 — Self-Refinement | 92.0% | 4.1 avg | High safety, expensive |
| 2 — Validation Loop | 92.0% | 2.2 avg | **Best overall** |
| 3 — RAG Injection | 46.0% | 1.0 avg | RAG causes distraction |

**Key finding:** Qualitative LLM-as-a-Judge evaluation is insufficient; objective multi-agent validation (Approach 2) is required to guarantee both mathematical accuracy and system efficiency.

See [`phase1/`](phase1/) for scripts and [`results/`](results/) for full benchmark data.

---

## Phase 1 Setup (Groq API)

In each Phase 1 script, replace:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

Then run in order:
```bash
python phase1/approach1_self_refine.py
python phase1/approach2_validation_loop.py
python phase1/approach3_rag_agent.py
python "phase1/Judge LLM.py"
```
