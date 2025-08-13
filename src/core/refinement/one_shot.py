import google.generativeai as genai
import csv
from pathlib import Path
from ...utils.config import get_api_key, GEMINI_MODEL


LOG_PATH = Path("refine-loop/logs/refine_outputs.csv")

q = "When did the first man land on Mars?" 

def call_gemini(prompt: str) -> str:
    return gemini_model.generate_content(prompt).text

def self_refine(prompt: str, max_passes: int = 3) -> list[dict]:
    history = []

    current_answer = call_gemini(prompt)

    for i in range(1, max_passes + 1):
        feedback_prompt = (
            f"Here's my answer to the question:\n\nQ: {prompt}\nA: {current_answer}\n\n"
            f"What is wrong or missing in this answer? Please critique it. If it's correct and complete, say so clearly."
        )
        feedback = call_gemini(feedback_prompt)

        # Check for stopping condition
        if "no issues" in feedback.lower() or "correct and complete" in feedback.lower():
            print(f"Stopping after {i} pass(es): Answer judged sufficient.\n")
            history.append({
                "iteration": i,
                "answer": current_answer,
                "feedback": feedback,
                "revised": current_answer  # No revision needed
            })
            break

        revision_prompt = (
            f"Question: {prompt}\n"
            f"Previous Answer: {current_answer}\n"
            f"Feedback: {feedback}\n\n"
            "Please provide an improved and corrected version."
        )
        revised = call_gemini(revision_prompt)

        history.append({
            "iteration": i,
            "answer": current_answer,
            "feedback": feedback,
            "revised": revised
        })

        current_answer = revised  # Continue loop with revised answer

    return history

def log_to_csv(prompt, ans1, feedback, ans2):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["prompt", "original_answer", "feedback", "revised_answer"]
    write_header = not LOG_PATH.exists()

    with open(LOG_PATH, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "prompt": prompt,
            "original_answer": ans1,
            "feedback": feedback,
            "revised_answer": ans2
        })

# Configure with your API key
api_key = get_api_key()
genai.configure(api_key=api_key)

# Load the model
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

if __name__ == "__main__":
    # q = "When did the Eiffel Tower open to the public?"
    results = self_refine(q, max_passes=3)

    for step in results:
        print(f"\n--- Iteration {step['iteration']} ---")
        print("Answer:", step['answer'])
        print("Feedback:", step['feedback'])
        print("Revised:", step['revised'])

        log_to_csv(q, step['answer'], step['feedback'], step['revised'])
