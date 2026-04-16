import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# ==========================================
# 1. API CONFIGURATION
# ==========================================
# Make sure to put your active Groq key here
GROQ_API_KEY = "YOUR_GROQ_API"
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
JUDGE_MODEL = "llama-3.3-70b-versatile"

# ==========================================
# 2. THE JUDGE PROMPT
# ==========================================
def evaluate_hint_with_judge(problem, correct_answer, student_error, hint_to_grade):
    system_prompt = """You are an expert educational researcher and behavioral evaluator. 
    Your task is to grade an AI-generated homework hint designed for a parent to read to their 7th-grade child.
    
    Evaluate the AI-Generated Hint on a scale of 1 to 5 across the three metrics below.
    
    METRIC 1: Answer Leakage (Safety)
    - Score 1: Total Leakage (Explicitly states the final correct number or exact formula).
    - Score 3: Partial Leakage (Gives away too much, leaving little thinking for the student).
    - Score 5: Perfect Safety (Completely protects the final answer).
    
    METRIC 2: Pedagogical Quality (Targeting)
    - Score 1: Generic/Useless ("Check your math").
    - Score 3: Helpful but blunt (Points out the error but no Socratic question).
    - Score 5: Perfect Socratic Scaffolding (Directly targets the logical flaw with a guiding question).
    
    METRIC 3: Caregiver Tone
    - Score 1: Robotic, overly complex, or condescending.
    - Score 3: Polite but unnatural or academic.
    - Score 5: Perfect Caregiver Persona (Warm, encouraging, sounds like a supportive parent).
    
    OUTPUT FORMAT: You must output ONLY a valid JSON object.
    {
      "leakage_score": <int 1-5>,
      "pedagogical_score": <int 1-5>,
      "tone_score": <int 1-5>
    }"""
    
    user_prompt = f"Problem: {problem}\nCorrect Answer: {correct_answer}\nStudent Error: {student_error}\nGenerated Hint to Grade: {hint_to_grade}"

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # Very low temp for consistent grading
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"leakage_score": 0, "pedagogical_score": 0, "tone_score": 0}

# ==========================================
# 3. ORCHESTRATOR & DATA MERGING
# ==========================================
print("Loading the 3 Result Datasets...")
df1 = pd.read_excel("D:\\NLP Applications Project\\Approach1_SelfRefine_Results.xlsx")
df2 = pd.read_excel("D:\\NLP Applications Project\\Approach2_ValidationLoop_Results.xlsx")
df3 = pd.read_excel("D:\\NLP Applications Project\\Approach3_ValidationLoop(KB)_Results.xlsx") # This is your App 3 RAG file

# Create a master list to hold the scores
master_results = []

print("Starting Blind Judge Evaluation (This will take a few minutes)...\n")

# Since the Original Problem, Correct Answer, and Wrong Answer are identical across all 3 files,
# we just loop through the index (0 to 49)
for index in range(len(df1)):
    print(f"⚖️ Grading Problem {index + 1}/50...")
    
    problem = df1.at[index, "Original Problem"]
    correct = df1.at[index, "Correct Answer"]
    error = df1.at[index, "Student Wrong Answer"]
    
    # Grab the final hints from each approach
    hint1 = df1.at[index, "Self_Refine_Final_Hint"]
    hint2 = df2.at[index, "Validation_Final_Hint"]
    hint3 = df3.at[index, "Last_Hint_Used"]
    
    # Grade Approach 1
    scores1 = evaluate_hint_with_judge(problem, correct, error, hint1)
    time.sleep(1) # Rate limit protection
    
    # Grade Approach 2
    scores2 = evaluate_hint_with_judge(problem, correct, error, hint2)
    time.sleep(1)
    
    # Grade Approach 3
    scores3 = evaluate_hint_with_judge(problem, correct, error, hint3)
    time.sleep(2)
    
    # Save the data
    master_results.append({"Approach": "1 (Self-Refine)", "Leakage": scores1['leakage_score'], "Pedagogy": scores1['pedagogical_score'], "Tone": scores1['tone_score']})
    master_results.append({"Approach": "2 (Validation Loop)", "Leakage": scores2['leakage_score'], "Pedagogy": scores2['pedagogical_score'], "Tone": scores2['tone_score']})
    master_results.append({"Approach": "3 (RAG Agent)", "Leakage": scores3['leakage_score'], "Pedagogy": scores3['pedagogical_score'], "Tone": scores3['tone_score']})

# ==========================================
# 4. SAVE AND VISUALIZE THE JUDGE SCORES
# ==========================================
print("\nEvaluation Complete! Generating Final Charts...")
results_df = pd.DataFrame(master_results)

# Filter out any API errors (score 0)
results_df = results_df[results_df['Leakage'] > 0]

# Calculate averages
avg_scores = results_df.groupby('Approach').mean().reset_index()
print(avg_scores)

# Save to Excel
avg_scores.to_excel("Final_Judge_Averages.xlsx", index=False)

# Plotting the Final Comparison
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics = ['Leakage', 'Pedagogy', 'Tone']
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, metric in enumerate(metrics):
    sns.barplot(data=avg_scores, x='Approach', y=metric, palette=colors, ax=axes[i])
    axes[i].set_title(f'Average {metric} Score (Out of 5)', fontsize=14)
    axes[i].set_ylim(0, 5.5)
    axes[i].set_ylabel("Score")
    axes[i].set_xticklabels(['App 1', 'App 2', 'App 3'])
    
    # Add number labels on top of bars
    for p in axes[i].patches:
        axes[i].annotate(f"{p.get_height():.2f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig("Qualitative_Judge_Scores.png", dpi=300)
print("\n🎉 Done! 'Qualitative_Judge_Scores.png' is saved to your folder.")
