import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ==========================================
# 1. API CONFIGURATION
# ==========================================
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Using Qwen 3 32B to completely eliminate Meta Llama Intra-Model Bias
JUDGE_MODEL = "qwen/qwen3-32b" 

# ==========================================
# 2. THE PAIRWISE SYSTEM PROMPT
# ==========================================
def evaluate_pairwise(problem, correct_answer, student_error, hint_a, hint_b):
    system_prompt = """You are an expert, highly critical educational researcher. 
    Your task is to compare two AI-generated math hints and determine which one is objectively better for a 7th-grade student.

    CRITERIA FOR WINNING:
    1. Safety (No Leakage): The hint MUST NOT give away the final correct number or do the arithmetic for the student.
    2. Pedagogy: The hint should directly target the student's specific logical flaw with a guiding Socratic question.
    3. Tone: The hint should sound warm, encouraging, and natural for a parent to read.

    INSTRUCTIONS:
    - You must choose strictly 'A' or 'B'. 
    - You are not allowed to declare a tie. 
    - Be ruthless. Even if both are good, pick the one that leaves more cognitive work for the student.

    OUTPUT FORMAT: You must output ONLY a valid JSON object.
    {
      "reasoning": "A 1-sentence explanation of why the winner is superior.",
      "winner": "A" or "B"
    }"""
    
    user_prompt = f"""
    Problem: {problem}
    Correct Answer: {correct_answer}
    Student Error: {student_error}
    
    Hint A: {hint_a}
    Hint B: {hint_b}
    """

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # Very low temp for consistent logic
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("winner", "Error")
    except Exception as e:
        print(f"Judge Error: {e}")
        return "Error"

# ==========================================
# 3. ORCHESTRATOR & DATA MERGING
# ==========================================
print("Loading the 3 Result Datasets...")
df1 = pd.read_excel("Results/Approach1_SelfRefine_Results.xlsx")
df2 = pd.read_excel("Results/Approach2_ValidationLoop_Results.xlsx")
df3 = pd.read_excel("Results/Phase1_Agentic_Results.xlsx") 

# Win counters
wins = {
    "Approach 1": 0,
    "Approach 2": 0,
    "Approach 3": 0
}

total_matchups_per_pair = len(df1) # 50 problems

print("Starting Pairwise Arena Matchups (This will take a few minutes)...\n")

for index in range(len(df1)):
    print(f"🥊 Matchups for Problem {index + 1}/50...")
    
    problem = df1.at[index, "Original Problem"]
    correct = df1.at[index, "Correct Answer"]
    error = df1.at[index, "Student Wrong Answer"]
    
    hint1 = df1.at[index, "Self_Refine_Final_Hint"]
    hint2 = df2.at[index, "Validation_Final_Hint"]
    hint3 = df3.at[index, "Last_Hint_Used"]
    
    # --- MATCHUP 1: Approach 1 vs Approach 2 ---
    # Hint A = App 1, Hint B = App 2
    winner_1v2 = evaluate_pairwise(problem, correct, error, hint1, hint2)
    if winner_1v2 == "A": wins["Approach 1"] += 1
    elif winner_1v2 == "B": wins["Approach 2"] += 1
    time.sleep(1) 
    
    # --- MATCHUP 2: Approach 2 vs Approach 3 ---
    # Hint A = App 2, Hint B = App 3
    winner_2v3 = evaluate_pairwise(problem, correct, error, hint2, hint3)
    if winner_2v3 == "A": wins["Approach 2"] += 1
    elif winner_2v3 == "B": wins["Approach 3"] += 1
    time.sleep(1)
    
    # --- MATCHUP 3: Approach 1 vs Approach 3 ---
    # Hint A = App 1, Hint B = App 3
    winner_1v3 = evaluate_pairwise(problem, correct, error, hint1, hint3)
    if winner_1v3 == "A": wins["Approach 1"] += 1
    elif winner_1v3 == "B": wins["Approach 3"] += 1
    time.sleep(1.5)

# ==========================================
# 4. CALCULATE WIN RATES & VISUALIZE
# ==========================================
print("\nEvaluation Complete! Calculating Win Rates...")

# Each approach competes in exactly 100 matchups total (50 against each of the other two approaches)
total_games_per_approach = total_matchups_per_pair * 2 

win_rates = {
    "App 1 (Self-Refine)": (wins["Approach 1"] / total_games_per_approach) * 100,
    "App 2 (Validation Loop)": (wins["Approach 2"] / total_games_per_approach) * 100,
    "App 3 (RAG Agent)": (wins["Approach 3"] / total_games_per_approach) * 100
}

print(f"Overall Win Rates:\n{win_rates}")

# Plotting the Final Win Rate Chart
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))

approaches = list(win_rates.keys())
rates = list(win_rates.values())
colors = ['#4C72B0', '#55A868', '#C44E52']

bars = plt.bar(approaches, rates, color=colors)
plt.title('Head-to-Head Pairwise Win Rate (%)', fontsize=15, pad=15)
plt.ylabel('Overall Win Rate (%)', fontsize=12)
plt.ylim(0, 100)

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", 
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("Pairwise_Win_Rates.png", dpi=300)
print("\n🎉 Done! 'Pairwise_Win_Rates.png' is saved to your folder.")
