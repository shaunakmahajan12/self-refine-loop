#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Setting up environment"
python -V
pip install -r config/requirements.txt >/dev/null

echo "[2/5] Generating refinement data (this uses GOOGLE_API_KEY)"
python - <<'PY'
from src.core.refinement.batch_self_refine import process_all_prompts
process_all_prompts(checkpoint_interval=25)
PY

echo "[3/5] Training improved SVM critic"
python src/core/critics/improved_svm_critic.py | sed -n '1,200p'

echo "[4/5] Evaluating BLEU/ROUGE on revisions"
python src/analysis/evaluate_metrics.py | sed -n '1,200p'

echo "[5/5] Running robust evaluation (CV + held-out + calibration)"
python src/analysis/robust_evaluation.py | sed -n '1,400p'

echo "Done. See logs/ for CSVs and figures."
