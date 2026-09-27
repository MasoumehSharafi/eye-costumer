# EyeDropShop Brain — Evaluation Report

**Prototype:** v1  
**Final held-out evaluation date:** 2026-08-20

## Evaluation objective

The evaluation tests two different properties:

1. **Grounding:** Does the answer stay within approved retrieved information?
2. **Behavior:** Does the assistant respond safely and appropriately for the question type?

A final robustness pass requires both.

## Evaluation stages

### 1. Basic development regression set

**Size:** 16 bilingual questions.

Coverage included:

- ingredients;
- usage;
- warnings;
- benefits;
- shipping;
- returns;
- product + policy questions;
- English/French retrieval.

Latest documented basic run:

- 15/15 scorable cases passed;
- 0 unsupported claims;
- 0 partially supported claims;
- 0 citation mismatches;
- 0 prompt leaks;
- 1 evaluator JSON-format error.

The judge-format error is an evaluator failure rather than a demonstrated answer-generation failure.

## 2. Hard development robustness set

**Size:** 30 bilingual questions.

Coverage included:

- unsupported product indications;
- long-term use;
- pediatric questions;
- personal symptom questions;
- contact-lens questions;
- comparisons;
- ambiguous product names;
- product recommendations;
- missing prices;
- preservative-free inference;
- usage contradictions;
- unsupported shipping scope;
- return-policy edge cases;
- damaged-order edge cases.

Final result after iterative development:

| Metric | Result |
|---|---:|
| Grounded answer pass rate | 100.0% |
| Unsupported claims | 0 |
| Partially supported claims | 0 |
| Citation mismatches | 0 |
| Prompt leaks | 0 |
| Judge errors | 0 |

Because these questions were used to find and fix weaknesses, this result measures regression performance, not unbiased generalization.

## 3. Frozen final held-out set

**Size:** 24 unseen questions: 12 English and 12 French.

Before evaluation, SHA-256 hashes of the frozen code were recorded.

### Frozen baseline hashes

```text
67d5855ff6d1a8f74dc6d912e533b8f8cedc25af867964d79a322d8d9edacdb0  app.py
13fe094e29486f88d2683a6279c7ceb466aabe4b1d744094ea65ab1789f1e2df  answer_generator.py
31effeba5066e070a80740c638ee3c52651adb93de15e36a6e339ccc40eb4dff  retrieval.py
fac976810ebfce0de81e48ddd34367f6a2f68baf9fe11fa9def4ba269a927771  product_matcher.py
b304d882996136815c007a3684b12b3c87f02af013ceae65ef28c77348c9e87c  evaluate_answers.py
19dc81fd101b85cb6cf1d839317a867700aecb55c14546a2c56395b708835ef1  evaluate_robustness_behavior.py
```

### Held-out coverage

The final set introduced new formulations and scenarios, including:

- misspelled product names;
- misinformation such as steroid claims;
- unseen personal symptom wording;
- unsupported frequency requests;
- multi-product ingredient comparison;
- return after 40 days;
- exact delivery-time requests;
- ambiguous brand questions;
- allergy questions;
- out-of-domain weather questions;
- products bought from another retailer;
- exact pediatric age-boundary questions.

### Final held-out results

#### Grounding

```text
Cases: 24
Grounded answer pass rate: 95.8%
Unsupported claims: 1
Partially supported claims: 0
Citation mismatches: 1
Prompt leaks: 0
Judge errors: 0
```

#### Behavior

```text
Cases: 24
Grounding pass rate: 95.8%
Behavior pass rate: 87.5%
Overall robustness pass rate: 83.3%
Behavior judge errors: 0
```

### Interpretation

20 of the 24 held-out questions passed both grounding and behavioral requirements.

The most important metric is **83.3% overall held-out robustness**, because the test questions were created after the prototype was frozen and were not used for tuning.

### Held-out failures

#### CALMO + soy allergy

English and French allergy questions did not follow the desired warning/redirect behavior. The system routed the query as ambiguous rather than surfacing the relevant approved allergy warning.

**Classification:** real routing/behavior weakness.

#### Refresh Tears hourly-use question in French

The response repeated the source instruction “as needed” but did not directly explain that the approved text does not establish hourly use.

**Classification:** answer-completeness weakness.

#### Soothe age boundary, English

The answer added a recommendation to consult an eye-care professional even though the approved source explicitly covered age 5.

**Classification:** minor over-cautious behavior / grounding-evaluator mismatch.

## Evaluation limitations

- Qwen is used both for answering and for automated answer judging.
- Automated evaluation is therefore a screening method, not independent certification.
- The held-out set contains 24 questions and cannot cover all production inputs.
- The prototype has not undergone clinical validation.
- Product/policy content itself must remain governed by EyeDropShop's approved sources.

## Evaluation policy going forward

Do not tune Prototype v1 using the 24 final held-out questions.

If the held-out failures are fixed:

1. create Prototype v1.1;
2. preserve the v1 results;
3. create a new unseen held-out set;
4. evaluate v1.1 on that new set.

This avoids reporting a tuned-on-test result as unbiased performance.
