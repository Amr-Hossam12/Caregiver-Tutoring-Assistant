import os
from openai import OpenAI
import pandas as pd


# ==========================================
# 1. API AND MODEL CONFIGURATION
# ==========================================

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", 
    api_key="YOUR_GROQ_API_KEY_HERE", 
)

# The Tutor stays the same (it's the newest, smartest 70B model!)
TUTOR_MODEL = "llama-3.3-70b-versatile"

# Swap the Student to the stable, incredibly fast Llama 3.1 8B model
STUDENT_MODEL = "llama-3.1-8b-instant"

# ==========================================
# 2. THE PIPELINE FUNCTIONS
# ==========================================

def generate_tutor_hint(problem, correct_solution, student_wrong_answer):
    """Stage 2: The Tutor Model generates a Socratic hint."""
    system_prompt = """You are an expert, encouraging middle school math tutor. 
    Your goal is to help a student fix their mistake without giving away the final answer. 
    Review the Problem, the Correct Solution (your ground truth), and the Student's Incorrect Logic.
    Generate a short, Socratic hint (1-3 sentences) that points the student toward their specific cognitive error."""

    user_input = f"""
    Problem: {problem}
    Correct Solution: {correct_solution}
    Student's Incorrect Logic: {student_wrong_answer}
    """

    response = client.chat.completions.create(
        model=TUTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7 # Slight creativity for generating a good pedagogical hint
    )
    return response.choices[0].message.content

def simulate_student_validation(problem, student_wrong_answer, tutor_hint):
    """Stage 3: The Student Model tests if the hint actually works."""
    system_prompt = """You are a middle school math student. You tried to solve a problem and got it wrong. 
    Your teacher just gave you a hint. 
    Read the original problem, your original wrong answer, and the teacher's hint. 
    Apply the hint to fix your logic. 
    CRITICAL INSTRUCTION: Your final output must end with the exact numerical answer. Do not write a long paragraph."""

    user_input = f"""
    Problem: {problem}
    My Original Wrong Answer: {student_wrong_answer}
    Teacher's Hint: {tutor_hint}
    """

    response = client.chat.completions.create(
        model=STUDENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1 # Very low temperature so the "student" focuses strictly on math calculation
    )
    return response.choices[0].message.content.strip()

def run_hint_pipeline(problem, correct_solution, student_wrong_answer, max_retries=3):
    """The orchestrator loop that ties the paper's architecture together."""
    print("Starting Cloud API Hint Pipeline...\n" + "="*40)
    
    # Extract the final correct number from the GSM8K format (e.g., "#### 17" -> "17")
    try:
        correct_final_number = correct_solution.split("#### ")[-1].strip()
    except IndexError:
        print("Error parsing correct solution format.")
        return None

    for attempt in range(1, max_retries + 1):
        print(f"\n--- ATTEMPT {attempt} ---")
        
        # Step A: Generate the Hint
        print("🤖 Tutor is thinking...")
        hint = generate_tutor_hint(problem, correct_solution, student_wrong_answer)
        print(f"💡 Generated Hint: {hint}\n")
        
        # Step B: Validate the Hint
        print("🧑‍🎓 Simulated Student is testing the hint...")
        simulated_student_answer = simulate_student_validation(problem, student_wrong_answer, hint)
        print(f"📝 Simulated Student's New Answer: {simulated_student_answer}\n")
        
        # Step C: Check for Success
        if correct_final_number in simulated_student_answer:
            print("✅ SUCCESS: The hint helped the student model solve the problem!")
            return hint # Return the validated hint
        else:
            print(f"❌ FAILED: The student didn't get '{correct_final_number}'. Regenerating hint...")
            
    print("\n⚠️ Max retries reached. Returning the safest generated hint from the final attempt.")
    return hint

# ==========================================
# 3. RUNNING A TEST (Main Execution)
# ==========================================


if __name__ == "__main__":
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_excel("Phase1_Caregiver_Dataset.xlsx") 
    
    # Create a list to store all the final hints
    final_hints_list = []

    # 2. Loop through every row
    for index, row in df.iterrows():
        print(f"\n" + "="*40)
        print(f"Processing Problem {index + 1} of {len(df)}")
        print("="*40)
        
        # We use str() on the correct answer just in case pandas reads it as an integer (like 18 instead of "18")
        # And we pass in the exact column names from your Excel file
        final_hint = run_hint_pipeline(
            problem=row['Original Problem'], 
            correct_solution=str(row['Correct Answer']), 
            student_wrong_answer=row['Student Wrong Answer'],
            max_retries=2
        )
        
        # Add the generated hint to our list
        final_hints_list.append(final_hint)

    # 3. Save the new results to a final Excel file (so it matches your input format!)
    df['Generated Tutor Hint'] = final_hints_list
    df.to_excel("Phase1_Benchmark_Results.xlsx", index=False)
    
    print("\n🎉 All done! Benchmark results saved to 'Phase1_Benchmark_Results.xlsx'")