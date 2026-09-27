# EyeDropShop Brain — Runbook

## 1. Requirements

- Python 3.11 environment;
- Ollama;
- `qwen3.5:9b`;
- Python dependencies from `requirements.txt`;
- local ChromaDB database.

## 2. Start Ollama

Check whether Ollama is available:

```bash
ollama list
```

If Python reports:

```text
ConnectionError: Failed to connect to Ollama
```

start the server:

```bash
ollama serve
```

Keep that terminal open.

## 3. Activate the environment

```bash
conda activate eyedropshop-brain
```

## 4. Start the application

From the project root:

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the address printed by Streamlit, typically:

```text
http://localhost:8501
```

Stop with:

```text
Ctrl+C
```

## 5. Basic retrieval regression test

```bash
python evaluate_retrieval.py
```

Expected development baseline:

```text
Exact product detection accuracy: 100.0%
Expected product hit@k: 100.0%
Expected section hit@k: 100.0%
Target retrieval hit@k: 100.0%
```

## 6. Basic answer regression test

```bash
python evaluate_answers.py
```

Review:

- unsupported claims;
- partially supported claims;
- citation mismatches;
- prompt leaks;
- judge errors.

A judge JSON-format error is an evaluator problem and should be re-run or inspected before being treated as a Brain failure.

## 7. Development robustness test

```bash
python evaluate_answers.py   --file evaluation/robustness_eval.csv   --csv-output evaluation/robustness_answer_results.csv   --json-output evaluation/robustness_answer_details.json
```

Then:

```bash
python evaluate_robustness_behavior.py
```

The frozen development baseline reached 100% grounding on this set after iterative development.

## 8. Final held-out evaluation

The final v1 held-out evaluation has already been run and should **not** be re-used for tuning.

The command was:

```bash
./run_final_holdout.sh
```

It:

1. hashes the frozen code;
2. runs grounding evaluation;
3. runs behavior evaluation;
4. saves all result files.

Final v1 held-out result:

```text
Grounding:          95.8%
Behavior:           87.5%
Overall robustness: 83.3%
```

## 9. Rebuild the product database

Only do this when the product dataset changes.

```bash
python build_product_database.py
python add_policies_to_database.py
```

Important: product database rebuild recreates the collection, so policy ingestion must be run again afterward.

## 10. Troubleshooting

### Ollama connection error

```text
ConnectionError: Failed to connect to Ollama
```

Fix:

```bash
ollama serve
```

### Hugging Face unauthenticated warning

The BGE-M3 model may print a warning that requests to Hugging Face are unauthenticated.

This does not stop local inference after the model is available. A Hugging Face token can be configured if higher Hub rate limits are needed.

### Transformers image-processor warnings

The environment may display warnings/errors referring to optional image processors from unrelated Transformers modules.

The prototype is text-only. Run Streamlit with:

```bash
streamlit run app.py --server.fileWatcherType none
```

to reduce optional module inspection.

## 11. Files that should not be deleted

Keep:

```text
prototype_data/products/products_30_bilingual.csv
chroma_products_db/
evaluation/final_baseline_sha256.txt
evaluation/final_holdout_eval.csv
evaluation/final_holdout_answer_results.csv
evaluation/final_holdout_answer_details.json
evaluation/final_holdout_behavior_results.csv
evaluation/final_holdout_behavior_details.json
```

These are needed to reproduce and document the delivered prototype.
