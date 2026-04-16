import pandas as pd
import json
import time
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# --- SECURITY WARNING: Replace with a NEW key, your old one was leaked! ---
GROQ_API_KEY = "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"
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

print("Loading embedding model (this takes a few seconds the first time)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 1. RAG Knowledge Base
knowledge_base = [
    {
        "error_type": "Order of Operations",
        "description": "Adding before multiplying, ignoring parentheses, or processing left-to-right incorrectly.",
        "strategy": "Ask the student to recite PEMDAS and identify which operation comes first in the specific equation before doing any math."
    },
    {
        "error_type": "Percentage and Ratio Mistakes",
        "description": "Applying a percentage to the wrong base number, or adding percentages directly.",
        "strategy": "Prompt the student to identify 'what is the 100% whole in this scenario?' before applying the percentage."
    },
    {
        "error_type": "Unit Conversion Failure",
        "description": "Mixing up units like inches/feet, hours/minutes, or forgetting to convert before dividing.",
        "strategy": "Ask the student to look at the labels on their numbers. Ask: 'Are these apples and oranges? How do we make the units match?'"
    },
    {
        "error_type": "Reading Comprehension / Missing Steps",
        "description": "Stopping halfway through a problem, answering the wrong question, or forgetting the final step.",
        "strategy": "Ask the student to re-read the very last sentence of the prompt out loud, and ask 'Does our current number answer this specific question?'"
    },
    {
        "error_type": "Basic Arithmetic Setup",
        "description": "Adding instead of multiplying, subtracting instead of adding, or setting up the equation backwards.",
        "strategy": "Ask the student to draw a quick picture or use small substitute numbers (like 2 and 3) to see if their chosen operation makes logical sense."
    }
]

kb_descriptions = [item["description"] for item in knowledge_base]
kb_embeddings = embedder.encode(kb_descriptions)

# 2. RAG Retrieval Function
def retrieve_strategy(caregiver_input):
    query_embedding = embedder.encode(caregiver_input)
    hits = util.semantic_search(query_embedding, kb_embeddings, top_k=1)
    best_match_idx = hits[0][0]['corpus_id']
    return knowledge_base[best_match_idx]['strategy']

# 3. Tutor LLM Function (Generates the Hint)
def generate_rag_hint(model_name, problem, correct_ans, caregiver_input, retrieved_strategy):
    system_prompt = f"""
    You are an expert, empathetic educational copilot for parents.
    CRITICAL TUTORING STRATEGY TO USE: {retrieved_strategy}

    Instructions:
    1. Do NOT reveal the final correct answer or the exact formula.
    2. Read the caregiver's input to understand the child's error.
    3. Apply the CRITICAL TUTORING STRATEGY provided above to formulate your hint.
    4. Use language appropriate for a parent speaking to a 7th grader.

    Output Format: You must output ONLY a valid JSON object with these two keys:
    {{
    "internal_reasoning": "Explain how you applied the required strategy.",
    "caregiver_hint": "The exact script the parent should read to the child."
    }}
    """
    user_prompt = f"Problem: {problem}\nCorrect Answer: {correct_ans}\nCaregiver says: {caregiver_input}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        # return both the hint AND the token count
        return result.get("caregiver_hint", "Error"), response.usage.total_tokens
    except Exception as e:
        print(f"Tutor Error: {e}")
        return "API Error", 0

# 4. Student LLM Function (Simulates the Child guessing again)
def simulate_student_response(model_name, problem, hint):
    system_prompt = """
    You are a 7th-grade student taking a math test. You previously got the wrong answer.
    Your parent just gave you a helpful hint.
    Use their hint to try and solve the problem again.

    Output Format: You must output ONLY a valid JSON object:
    {
        "thought_process": "How the hint changes your math steps",
        "final_number": "Only the final numeric answer you calculated (e.g., '15', '4.5')"
    }
    """
    user_prompt = f"Problem: {problem}\nParent's Hint: {hint}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        result = json.loads(response.choices[0].message.content)
        # return both the answer AND the token count
        return str(result.get("final_number", "")).strip(), response.usage.total_tokens
    except Exception as e:
        print(f"Student Error: {e}")
        return "Error", 0

# Helper function to clean numbers for accurate comparison
def check_answer_match(student_ans, correct_ans):
    s_clean = re.sub(r'[^\d.]', '', str(student_ans))
    c_clean = re.sub(r'[^\d.]', '', str(correct_ans))
    try:
        return float(s_clean) == float(c_clean)
    except:
        return s_clean == c_clean

# 5. Execute the Simulation Loop
print("Loading dataset...")
df = pd.read_excel("Phase1_Caregiver_Dataset.xlsx")

TUTOR_MODEL = "llama-3.3-70b-versatile"
STUDENT_MODEL = "llama-3.1-8b-instant"

# Setup the exact columns you asked for
df["Simulation_Status"] = ""       # Pass/Fail column
df["Attempts_Taken"] = 0
df["Last_Hint_Used"] = ""
df["Last_Strategy_Used"] = ""
df["Total_API_Calls"] = 0          # NEW: API Tracker
df["Total_Tokens_Used"] = 0        # NEW: Token Tracker

print("Starting Multi-Agent Simulation Benchmark...")

for index, row in df.iterrows():
    print(f"\nProcessing Row {index + 1}/50...")

    problem = row["Original Problem"]
    correct_ans = row["Correct Answer"]
    current_caregiver_input = row["Caregiver Input"]

    # Initialize tracking variables for this specific row
    problem_api_calls = 0
    problem_total_tokens = 0

    max_attempts = 2
    status = "Failed" # Defaults to failed
    last_hint = ""
    last_strategy = ""

    # The 2-Attempt Loop
    for attempt in range(1, max_attempts + 1):
        print(f"  Attempt {attempt}...")

        # Step A: Tutor figures out strategy and gives hint
        strategy = retrieve_strategy(current_caregiver_input) # Embedder runs locally (0 tokens)

        time.sleep(2) # Rate limit pause
        hint, t_tokens = generate_rag_hint(TUTOR_MODEL, problem, correct_ans, current_caregiver_input, strategy)

        problem_api_calls += 1
        problem_total_tokens += t_tokens

        last_hint = hint
        last_strategy = strategy

        # Step B: Student tries to use the hint
        time.sleep(2) # Rate limit pause
        student_new_answer, s_tokens = simulate_student_response(STUDENT_MODEL, problem, hint)

        problem_api_calls += 1
        problem_total_tokens += s_tokens

        # Step C: Evaluate Student's Answer
        if check_answer_match(student_new_answer, correct_ans):
            status = "Success" # Marks Pass/Fail correctly
            print(f"    -> Student got it right! ({student_new_answer})")
            break # Exit the loop early since they succeeded!
        else:
            print(f"    -> Student failed again. Guessed: {student_new_answer}")
            current_caregiver_input = f"I gave them your hint, but they tried again and got {student_new_answer}. They are still confused. Can you give a totally different hint?"

    time.sleep(3) # Rest between rows to protect rate limit

    # Save the final results for this row
    df.at[index, "Simulation_Status"] = status
    df.at[index, "Attempts_Taken"] = attempt
    df.at[index, "Last_Hint_Used"] = last_hint
    df.at[index, "Last_Strategy_Used"] = last_strategy
    df.at[index, "Total_API_Calls"] = problem_api_calls       # SAVES API CALLS
    df.at[index, "Total_Tokens_Used"] = problem_total_tokens  # SAVES TOKENS

# 6. Save the final output
output_file = "Approach3_ValidationLoop(KB)_Results.xlsx"
df.to_excel(output_file, index=False)
print(f"\nSuccess! Agentic Simulation Results saved to {output_file}.")