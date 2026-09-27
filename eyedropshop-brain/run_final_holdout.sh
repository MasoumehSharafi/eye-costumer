#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p evaluation

echo "============================================================"
echo "1) Freeze current prototype baseline"
echo "============================================================"

shasum -a 256 \
  app.py \
  answer_generator.py \
  retrieval.py \
  product_matcher.py \
  evaluate_answers.py \
  evaluate_robustness_behavior.py \
  > evaluation/final_baseline_sha256.txt

cat evaluation/final_baseline_sha256.txt

echo
echo "============================================================"
echo "2) Final held-out grounding evaluation"
echo "============================================================"

python evaluate_answers.py \
  --file evaluation/final_holdout_eval.csv \
  --csv-output evaluation/final_holdout_answer_results.csv \
  --json-output evaluation/final_holdout_answer_details.json

echo
echo "============================================================"
echo "3) Final held-out behavior evaluation"
echo "============================================================"

python evaluate_robustness_behavior.py \
  --cases evaluation/final_holdout_eval.csv \
  --answers evaluation/final_holdout_answer_results.csv \
  --output evaluation/final_holdout_behavior_results.csv \
  --details evaluation/final_holdout_behavior_details.json

echo
echo "============================================================"
echo "FINAL HOLD-OUT EVALUATION COMPLETE"
echo "============================================================"
echo "Do not modify the model/code based on these held-out questions."
