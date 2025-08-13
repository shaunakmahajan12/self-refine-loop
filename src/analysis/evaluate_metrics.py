import pandas as pd
from evaluate import load
import os

# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Load cleaned CSV (make sure you update path if needed)
df = pd.read_csv("logs/refine_outputs.csv")

# Load evaluation metrics
bleu = load("bleu")
rouge = load("rouge")

# Initialize storage
bleu_scores, rouge_scores = [], []

# Iterate over all rows
for _, row in df.iterrows():
    ref = [row["revised_answer"].strip()]
    hyp = row["original_answer"].strip()

    # BLEU expects list of predictions and list of list of references
    bleu_score = bleu.compute(predictions=[hyp], references=[[ref[0]]])["bleu"]
    rouge_score = rouge.compute(predictions=[hyp], references=ref)["rougeL"]

    bleu_scores.append(bleu_score)
    rouge_scores.append(rouge_score)

# Add metrics to DataFrame
df["BLEU_Original_vs_Revised"] = bleu_scores
df["ROUGE-L_Original_vs_Revised"] = rouge_scores

# Save to logs folder
output_path = "logs/evaluated_outputs.csv"
df.to_csv(output_path, index=False)

print(f"✅ BLEU and ROUGE evaluation complete. Results saved to:\n{output_path}")
