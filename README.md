# Pedagogical Math Hint Generation: Evaluating Agentic Architectures

## 1. Project Abstract
When caregivers use Large Language Models (LLMs) to help children with math homework, standard Zero-Shot prompting frequently results in "Answer Leakage"—giving away the final answer rather than providing Socratic scaffolding. 

This project tackles this problem by implementing and benchmarking three advanced NLP architectures based on 2023–2025 research. We evaluate **Internal Self-Refinement**, **Multi-Agent Validation**, and **Retrieval-Augmented Generation (RAG)** to determine which architecture produces the safest and most effective educational hints.

## 2. Setup & Execution Instructions

### Prerequisites
1. Ensure you have Python 3.10+ installed.
2. Obtain a free API key from [Groq Cloud](https://console.groq.com/).

### Installation
Clone this repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
In each of the Python scripts, locate the API configuration block at the top and insert your Groq API key:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

### How to Run
The pipeline is divided into three standalone scripts and one evaluation script. Run them in the following order. *(Note: The scripts include built-in rate-limit protections for the Groq API).*

1. `python approach1_self_refine.py`
2. `python approach2_validation_loop.py`
3. `python approach3_rag_agent.py` 
4. `python evaluate_judge.py` *(Generates final visual charts)*

## 3. The Three Methodologies

To establish a baseline, we assume standard **Zero-Shot Generation** as the control. We then evaluated three distinct architectures:

* **Approach 1: Dynamic Self-Refinement** (Based on *Madaan et al., 2023*)
  * **Concept:** The Llama-3.3-70B Teacher model generates a draft, critiques its own pedagogy, and rewrites the hint. It utilizes a dynamic stop condition to break the loop early if it scores itself a perfect "5/5".
* **Approach 2: Simulated Student Validation Loop** (Based on *Tonga et al., 2025*)
  * **Concept:** The 70B Teacher generates a hint and passes it to a weaker Llama-3.1-8B "Student" model. If the student fails to calculate the correct math answer using the hint, the Teacher is forced to rewrite it.
* **Approach 3: Personalized RAG Injection** (Based on *Liu et al., 2025*)
  * **Concept:** Utilizing `sentence-transformers`, we map the student's specific cognitive error (e.g., "Unit Conversion Failure") to a local vector database of proven pedagogical strategies, forcing the LLM to follow strict educational theory.

## 4. Evaluation Metrics Defined

We evaluated the systems using both Quantitative System Metrics and Qualitative LLM-as-a-Judge grading.

* **Student Pass Rate (Math Accuracy):** The percentage of times the Simulated Student successfully reached the correct numerical answer after reading the final hint.
* **API Cost & Latency:** Tracking the total tokens generated and the number of API calls required to finalize a single math problem.
* **Answer Leakage:** Scored 1-5 by the LLM Judge on how safely the hint protected the final answer.
* **Pedagogy & Tone:** Scored 1-5 by the LLM Judge on Socratic helpfulness and caregiver warmth.

## 5. Conclusion & Discussion of Trade-offs

Our benchmark yielded two major discoveries regarding the trade-offs in modern NLP systems:

1. **The Cost vs. Safety Trade-off:** Approach 1 (Self-Refine) and Approach 2 (Validation Loop) both achieved an exceptional **92.0% Student Pass Rate**. However, Approach 1 requires nearly double the API calls (4.1 vs 2.2) and tokens. Therefore, Approach 2's Multi-Agent Validation is the objectively superior architecture for a production environment.
2. **The Ceiling Effect & RAG Distraction:** Our qualitative LLM-as-a-Judge evaluation (using Llama 70B to grade Llama 70B) suffered from severe **Intra-Model Bias**, giving nearly all approaches a perfect 5.0/5.0. It confidently rated Approach 3 (RAG) as perfectly pedagogical. However, our simulated student **failed 54% of the time** when using those RAG hints. This proves that rigidly constraining an LLM with external RAG rules can cause a "distraction effect," breaking the model's fluid reasoning. 

**Final Verdict:** Qualitative LLM evaluation is insufficient for educational AI; objective multi-agent validation (Approach 2) is required to guarantee both mathematical accuracy and system efficiency.
