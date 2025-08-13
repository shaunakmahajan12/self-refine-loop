import pandas as pd
from pathlib import Path
import google.generativeai as genai
import time
from datetime import datetime

# Get script directory for path resolution
script_dir = Path(__file__).parent

# Read API key from file
with open(script_dir.parent.parent.parent / "config" / "API_KEY.txt", "r") as f:
    api_key = f.read().strip()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def save_checkpoint(rows, checkpoint_path):
    """Save current progress to a checkpoint file"""
    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

def load_checkpoint(checkpoint_path):
    """Load progress from checkpoint if it exists"""
    if checkpoint_path.exists():
        return pd.read_csv(checkpoint_path).to_dict('records')
    return []

def process_all_prompts(checkpoint_interval=10):
    """
    Process all prompts from the dataset with checkpointing.
    Args:
        checkpoint_interval: How often to save progress (default: every 10 prompts)
    """
    def call_gemini(prompt, retries=3, delay=5):
        for i in range(retries):
            try:
                return model.generate_content(prompt).text.strip()
            except Exception as e:
                if "429" in str(e):
                    wait = delay + i * 2
                    time.sleep(wait)
                else:
                    return "[ERROR]"
        return "[ERROR - too many retries]"

    df = pd.read_csv(
        script_dir.parent.parent.parent / "data" / "prompts.csv",
        quoting=1
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = script_dir.parent.parent.parent / "logs" / f"checkpoint_{timestamp}.csv"
    output_path = script_dir.parent.parent.parent / "logs" / f"refine_outputs_{timestamp}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_checkpoint(checkpoint_path)
    start_index = len(rows)

    start_time = time.time()
    last_checkpoint_time = start_time

    for i, row in df.iloc[start_index:].iterrows():
        prompt = row["prompt"]
        current_index = i + 1
        
        elapsed_time = time.time() - start_time
        avg_time_per_prompt = elapsed_time / (current_index - start_index) if current_index > start_index else 0
        estimated_remaining = avg_time_per_prompt * (len(df) - current_index)

        try:
            # Step 1: Initial Answer
            initial_prompt = (
                f"Please provide a clear and accurate answer to the following question.\n\n"
                f"Question: {prompt}"
            )
            original_answer = call_gemini(initial_prompt)
            time.sleep(1)

            # Step 2: Simple Feedback
            feedback_prompt = (
                f"Here's an answer to the question:\n\n"
                f"Q: {prompt}\n"
                f"A: {original_answer}\n\n"
                f"Please evaluate if this answer is complete and accurate. "
                f"If it needs improvement, explain why. If it's good, say so."
            )
            feedback = call_gemini(feedback_prompt)
            time.sleep(1)

            # Step 3: Determine if revision is needed
            needs_improvement = 1
            if any(phrase in feedback.lower() for phrase in [
                "good answer",
                "complete and accurate",
                "no improvement needed",
                "well answered",
                "satisfactory"
            ]):
                needs_improvement = 0
                revised_answer = original_answer
            else:
                # Simple revision prompt
                revision_prompt = (
                    f"Question: {prompt}\n"
                    f"Previous Answer: {original_answer}\n"
                    f"Feedback: {feedback}\n\n"
                    f"Please provide an improved version of the answer."
                )
                revised_answer = call_gemini(revision_prompt)
                time.sleep(1)

            rows.append({
                "prompt": prompt,
                "original_answer": original_answer,
                "feedback": feedback,
                "revised_answer": revised_answer,
                "needs_improvement": needs_improvement
            })

            # Save checkpoint periodically
            if (current_index - start_index) % checkpoint_interval == 0:
                save_checkpoint(rows, checkpoint_path)
                last_checkpoint_time = time.time()

        except Exception as e:
            print(f"❌ Error processing prompt {current_index}: {str(e)}")
            save_checkpoint(rows, checkpoint_path)
            raise

        # Optional: slow down to avoid burst limits
        time.sleep(1.5)

    # Save final results
    pd.DataFrame(rows).to_csv(output_path, index=False)
    total_time = time.time() - start_time
    print(f"\n✅ Successfully processed all {len(rows)} prompts!")
    print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    print(f"📊 Average time per prompt: {total_time/len(rows):.1f} seconds")
    print(f"💾 Final results saved to: {output_path}")

if __name__ == "__main__":
    process_all_prompts()
