import os
import pandas as pd
from openai import OpenAI

# ==========================================
# 1. API AND MODEL CONFIGURATION
# ==========================================

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="YOUR_GROQ_API_KEY_HERE", # Replace with your actual key
)

TUTOR_MODEL = "llama-3.3-70b-versatile"
STUDENT_MODEL = "llama-3.1-8b-instant"

# ==========================================
# 2. THE PROMPTS AND FUNCTIONS
# ==========================================

def generate_initial_hint(problem, correct_solution, student_wrong_answer):
    """Step 1: The Tutor generates the first draft of the hint."""
    system_prompt = """You are an expert, encouraging middle school math tutor.
    Review the Problem, the Correct Solution (your ground truth), and the Student's Incorrect Logic.
    Generate a short, Socratic hint (1-3 sentences) that points the student toward their specific cognitive error.
    Do NOT give away the final answer."""

    user_input = f"Problem: {problem}\nCorrect Solution: {correct_solution}\nStudent's Logic: {student_wrong_answer}"

    response = client.chat.completions.create(
        model=TUTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7
    )

    # Return both the text and the token count
    return response.choices[0].message.content, response.usage.total_tokens

def critique_hint(problem, correct_solution, student_wrong_answer, current_hint):
    """Step 2: The Tutor critiques its own draft."""
    system_prompt = """You are evaluating a draft of a math homework hint.
    Review the draft against two strict rules:
    1. Does it accidentally give away the final answer or exact calculation?
    2. Is it generic, or does it correctly target the student's specific cognitive error?

    INSTRUCTIONS:
    If the draft violates EITHER of these rules, write a 1-sentence critique of what must be fixed.
    If the draft is completely safe and pedagogically sound, you must output exactly the word: PERFECT. Do not write anything else."""

    user_input = f"Problem: {problem}\nCorrect Solution: {correct_solution}\nStudent's Logic: {student_wrong_answer}\n\nDraft Hint to Critique: {current_hint}"

    response = client.chat.completions.create(
        model=TUTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip(), response.usage.total_tokens

def refine_hint(current_hint, critique):
    """Step 3: The Tutor rewrites the hint based on its own critique."""
    system_prompt = """You are a tutor revising a homework hint.
    Read the original draft and the critique.
    Rewrite the draft hint so that it fixes the issue mentioned in the critique. Keep it to 1-3 sentences and Socratic."""

    user_input = f"Original Draft: {current_hint}\nCritique to Address: {critique}"

    response = client.chat.completions.create(
        model=TUTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content, response.usage.total_tokens

def student_final_exam(problem, student_wrong_answer, final_hint):
    """Step 4: The Student Agent takes the final exam to see if the hint actually works."""
    system_prompt = """You are a middle school math student. You tried to solve a problem and got it wrong.
    Your teacher just gave you a final hint.
    Apply the hint to fix your logic.
    CRITICAL INSTRUCTION: Your final output must end with the exact numerical answer. Do not write a long paragraph."""

    user_input = f"Problem: {problem}\nMy Original Wrong Answer: {student_wrong_answer}\nTeacher's Hint: {final_hint}"

    response = client.chat.completions.create(
        model=STUDENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip(), response.usage.total_tokens

# ==========================================
# 3. THE PIPELINE ORCHESTRATOR
# ==========================================

def run_self_refine_pipeline(problem, correct_solution, student_wrong_answer, max_loops=2):
    print("Starting Approach 1: Self-Refine Pipeline...\n" + "-"*40)

    # Initialize metric trackers for this specific problem
    problem_api_calls = 0
    problem_total_tokens = 0

    # Extract the final correct number
    correct_final_number = str(correct_solution).split("#### ")[-1].strip()

    # --- Step 1: Generate Initial Draft ---
    current_hint, tokens = generate_initial_hint(problem, correct_solution, student_wrong_answer)
    problem_api_calls += 1
    problem_total_tokens += tokens
    print(f"📝 Draft 1: {current_hint}")

    # --- Step 2: The Self-Critique Loop ---
    loops_taken = 0
    for attempt in range(max_loops):
        print(f"\n🔍 Self-Critique Loop {attempt + 1}/{max_loops}...")
        critique, crit_tokens = critique_hint(problem, correct_solution, student_wrong_answer, current_hint)

        problem_api_calls += 1
        problem_total_tokens += crit_tokens

        if "PERFECT" in critique.upper():
            print("✅ Tutor declared the hint PERFECT! Breaking loop early.")
            break
        else:
            print(f"⚠️ Flaw Found: {critique}")
            current_hint, ref_tokens = refine_hint(current_hint, critique)

            problem_api_calls += 1
            problem_total_tokens += ref_tokens
            loops_taken += 1
            print(f"🔄 Revised Hint: {current_hint}")

    # --- Step 3: The Final Exam (Student Agent) ---
    print("\n🧑‍🎓 Handing Final Hint to Student Agent for Exam...")
    student_answer, student_tokens = student_final_exam(problem, student_wrong_answer, current_hint)

    problem_api_calls += 1
    problem_total_tokens += student_tokens

    print(f"Student Output: {student_answer}")

    passed = "Yes" if correct_final_number in student_answer else "No"
    print(f"Student Passed? {passed}")
    print(f"System Cost: {problem_api_calls} API Calls | {problem_total_tokens} Tokens\n" + "="*40)

    # Return the data payload
    return current_hint, loops_taken, passed, problem_api_calls, problem_total_tokens

# ==========================================
# 4. MAIN EXECUTION & EXCEL EXPORT
# ==========================================

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_excel("Phase1_Caregiver_Dataset.xlsx")

    # Create lists to hold our new data
    final_hints = []
    loops_counts = []
    student_results = []
    api_call_totals = []
    token_totals = []

    # Loop through the dataset
    for index, row in df.iterrows():
        print(f"\nProcessing Problem {index + 1} of {len(df)}")

        hint, loops, passed, api_calls, tokens = run_self_refine_pipeline(
            problem=row['Original Problem'],
            correct_solution=row['Correct Answer'],
            student_wrong_answer=row['Student Wrong Answer'],
            max_loops=2 # THIS IS YOUR HYPERPARAMETER
        )

        final_hints.append(hint)
        loops_counts.append(loops)
        student_results.append(passed)
        api_call_totals.append(api_calls)
        token_totals.append(tokens)

    # Save to Excel
    df['Self_Refine_Final_Hint'] = final_hints
    df['Self_Refine_Loops_Taken'] = loops_counts
    df['Student_Passed_Exam'] = student_results
    df['Total_API_Calls'] = api_call_totals
    df['Total_Tokens_Used'] = token_totals

    output_filename = "Approach1_SelfRefine_Results.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"\n🎉 Success! All data saved to {output_filename}")