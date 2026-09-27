# EyeDropShop Brain — Delivery Summary

**Prototype:** v1  
**Date:** 2026-08-20  
**Scope:** local English/French EyeDropShop product-information prototype for Canada.

## What was delivered

A working local RAG prototype that:

- retrieves approved bilingual EyeDropShop product and policy content;
- uses BGE-M3 embeddings and ChromaDB;
- uses local Qwen3.5 9B through Ollama;
- provides a Streamlit interface;
- performs exact product matching, ambiguity detection, section-aware retrieval, and multi-product retrieval;
- cites retrieved sources;
- includes medical-safety and unsupported-information guards;
- supports English and French.

## Prototype data

- 284 unique English active products were identified in the source export.
- 233 have at least some French content.
- 54 met the bilingual prototype-field requirements.
- 30 products were selected for Prototype v1.
- The fixed prototype contains 60 language records.
- Canadian English/French shipping and return policies are included.
- FAQs are intentionally excluded pending source-quality and policy-conflict resolution.

## Evaluation result

### Development regression

The hard 30-question development set reached:

```text
Grounding: 100%
Unsupported claims: 0
Partial claims: 0
Citation mismatches: 0
Prompt leaks: 0
Judge errors: 0
```

This set was used iteratively and is therefore a regression result rather than an unbiased final estimate.

### Final frozen held-out evaluation

A new 24-question set was created after the code was frozen.

```text
Grounding pass rate:          95.8%
Behavior pass rate:           87.5%
Overall robustness pass rate: 83.3%
Prompt leaks:                  0
Judge errors:                  0
```

20 of 24 unseen questions passed both grounding and behavior.

## Main remaining limitations

- only 30 products are included;
- source data is static rather than live;
- French catalog coverage is incomplete;
- FAQ content is not yet approved for ingestion;
- the final held-out set exposed remaining allergy-routing, unsupported-frequency, and age-boundary behavior weaknesses;
- automated evaluation uses Qwen as judge and is not independent certification;
- the prototype has not undergone clinical validation;
- the interface is not production-authenticated.

## Recommended next release

Treat improvements as **Prototype v1.1**.

Priority work:

1. fix CALMO allergy routing;
2. improve direct handling of strict frequency questions;
3. refine age-boundary safety behavior;
4. create a new unseen test set;
5. expand the product catalog;
6. connect read-only live product data;
7. resolve FAQ source issues before ingestion.

## Delivery acceptance checklist

- [ ] `app.py` launches successfully.
- [ ] Ollama has `qwen3.5:9b`.
- [ ] `requirements.txt` installs successfully.
- [ ] ChromaDB is present.
- [ ] 30-product bilingual prototype dataset is present.
- [ ] English/French shipping and return policy content is present.
- [ ] Basic retrieval regression can run.
- [ ] Evaluation result files are included.
- [ ] Frozen SHA-256 baseline file is included.
- [ ] README and docs are included.
- [ ] Qwen model weights are not packaged.
