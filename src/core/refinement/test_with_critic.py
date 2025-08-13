import joblib
import pandas as pd
import google.generativeai as genai
import time
import os
import numpy as np
from pathlib import Path
import re
from google.api_core import retry
import random

# Get script directory for path resolution
script_dir = Path(__file__).parent

from ...utils.config import get_api_key, GEMINI_MODEL

# Load API key and configure Gemini
api_key = get_api_key()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(GEMINI_MODEL)

# Load improved SVM critic
from ..critics.improved_svm_critic import ImprovedSVMCritic
critic = ImprovedSVMCritic.load_model(script_dir.parent / "models")
vectorizer = critic.vectorizer
text_extractor = critic.text_extractor

def exponential_backoff(attempt):
    """Calculate backoff time with jitter."""
    base_delay = 2 ** attempt  # exponential backoff
    jitter = random.uniform(0, 0.1 * base_delay)  # add 10% jitter
    return min(base_delay + jitter, 120)  # cap at 120 seconds

# Function to call Gemini with retry logic
def call_gemini(prompt: str, max_retries: int = 8) -> str:
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                wait_time = exponential_backoff(attempt)
                print(f"\n⚠️ Rate limit hit. Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
                continue
            elif "ResourceExhausted" in str(e) and attempt < max_retries - 1:
                wait_time = 60  # Wait a full minute for quota reset
                print(f"\n⚠️ Quota exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            raise  # Re-raise if it's not a rate limit error or we're out of retries

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep important ones for code
    text = re.sub(r'[^\w\s\`\#\-\*\.]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(answer):
    # Preprocess text
    clean_text = preprocess_text(answer)
    
    # Create a dummy feedback (empty string) to match training structure
    dummy_feedback = ""
    clean_feedback = preprocess_text(dummy_feedback)
    
    # Extract numeric features for both answer and feedback
    answer_features = text_extractor.transform(pd.Series([answer]))
    feedback_features = text_extractor.transform(pd.Series([dummy_feedback]))
    
    # Fill any NaN values with 0
    answer_features = answer_features.fillna(0)
    feedback_features = feedback_features.fillna(0)
    
    # Calculate relative features (using same answer for both original and revised)
    relative_features = pd.DataFrame()
    for col in answer_features.columns:
        # Use a small epsilon to avoid division by zero
        relative_features[f'relative_{col}'] = 0  # No change since it's the same answer
    
    # Combine all features like in training
    X_numeric = pd.concat([
        answer_features.add_prefix('original_'),
        answer_features.add_prefix('revised_'),
        feedback_features.add_prefix('feedback_'),
        relative_features
    ], axis=1)
    
    # Fill any remaining NaN values with 0
    X_numeric = X_numeric.fillna(0)
    
    # Transform text features
    text_features = vectorizer.transform([clean_text + ' ' + clean_text + ' ' + clean_feedback])
    
    # Combine features
    combined_features = np.hstack([text_features.toarray(), X_numeric])
    
    # Final check for any NaN values
    if np.isnan(combined_features).any():
        combined_features = np.nan_to_num(combined_features, nan=0.0)
    
    return combined_features

# Core refine loop with external critic
def self_refine_with_external_critic(prompt: str, max_passes: int = 3) -> list[dict]:
    history = []
    current_answer = call_gemini(prompt)
    time.sleep(2)  # Add delay between initial answer and refinement

    for i in range(1, max_passes + 1):
        # Use external critic
        features_vec = extract_features(current_answer)
        critic_prediction = critic.predict(features_vec)[0]

        if critic_prediction == 0:
            print(f"✅ Critic judged answer sufficient at pass {i}.")
            history.append({
                "iteration": i,
                "prompt": prompt,
                "answer": current_answer,
                "feedback": "Critic: Answer is sufficient.",
                "revised": current_answer
            })
            break

        # Otherwise, ask Gemini to revise
        refinement_prompt = (
            f"Question: {prompt}\n"
            f"Previous Answer: {current_answer}\n"
            "Please revise and improve the answer without additional commentary."
        )
        revised_answer = call_gemini(refinement_prompt)
        time.sleep(3)  # Add longer delay between refinements

        history.append({
            "iteration": i,
            "prompt": prompt,
            "answer": current_answer,
            "feedback": "Critic: Answer needs improvement.",
            "revised": revised_answer
        })

        current_answer = revised_answer

    return history


# 🔍 Testing on prompts from CSV file
test_prompts_df = pd.read_csv(script_dir.parent / "data" / "test_prompts.csv")
test_prompts = test_prompts_df['prompt'].tolist()

all_results = []

for i, prompt in enumerate(test_prompts):
    print("\n====================")
    print(f"🔍 Processing prompt {i+1}/{len(test_prompts)}")
    print("Prompt:", prompt)
    
    try:
        refinement_results = self_refine_with_external_critic(prompt)
        for step in refinement_results:
            print("\n--- Iteration", step['iteration'], "---")
            print("Answer:", step['answer'])
            print("Feedback:", step['feedback'])
            print("Revised:", step['revised'])
            all_results.append(step)
        
        # Add delay between prompts to avoid rate limits
        if i < len(test_prompts) - 1:  # Don't delay after the last prompt
            delay = 5  # 5 second delay between prompts
            print(f"\nWaiting {delay} seconds before next prompt...")
            time.sleep(delay)
            
    except Exception as e:
        print(f"\n❌ Error processing prompt: {str(e)}")
        print("Saving results so far and exiting...")
        break

# 🔧 Save results to logs

df_results = pd.DataFrame(all_results)
logs_dir = script_dir.parent / "logs"
logs_dir.mkdir(exist_ok=True)
output_path = logs_dir / "test_refine_outputs.csv"
df_results.to_csv(output_path, index=False)
print(f"\n✅ All refinement results saved to:\n{output_path}")