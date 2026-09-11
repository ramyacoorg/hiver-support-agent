# AppleSupport AI Support Agent — Hiver SDE Intern Assignment

An AI agent for AppleSupport's real Twitter customer support: classifies
intent, drafts a grounded reply, and decides whether to auto-handle or
escalate to a human, with a stated reason.

Built on the real Kaggle "Customer Support on Twitter" dataset — not a
synthetic stand-in. See `report.md` for full results and `decision_log.md`
for the reasoning behind every non-obvious choice.

## Quickstart (reproduces headline results in well under 15 min)

```bash
pip install -r requirements.txt

# 1. Train on the 3,000-thread training pool, predict on the real 196-example
#    golden set (these are disjoint sets -- see report.md Section 2 for why):
python src/pipeline.py --train data/brand_threads_train.csv \
    --predict-on eval/golden_set.csv --out data/pipeline_output.csv

# 2. Score the real model against the golden set:
python eval/eval_harness.py --golden eval/golden_set.csv --predictions data/pipeline_output.csv

# 3. Compare against the trivial and keyword baselines:
python eval/baselines.py --golden eval/golden_set.csv
```

Expect: intent_accuracy≈0.265, routing_accuracy≈0.337 for the real model;
trivial baseline routing_accuracy≈0.745; keyword baseline intent_accuracy≈0.776.
**The real model loses to both baselines on headline accuracy — this is the
real, reproducible result, explained in report.md Sections 4-5, not a bug.**

Optional: set `ANTHROPIC_API_KEY` to also enable LLM-grounded reply drafting
and LLM-as-judge scoring (`src/grounding.py`, `eval/eval_harness.py`) — not
used for the headline numbers above.

## Repo layout

```
data/
  README.md                     how the real dataset was obtained + filtered
  sample_brand_tweets.csv       tiny synthetic set, zero-dependency code smoke test
  apple_filtered_clean.csv      real data, cleaned (~187K rows, AppleSupport-relevant)
  brand_threads_full.csv        all 69,389 real reconstructed thread pairs
  brand_threads_full_labelled.csv  same, with bootstrap intent/routing labels attached
  brand_threads_train.csv       3,000-thread training subsample (disjoint from golden set)
  pipeline_output.csv           (generated) model predictions on the golden set
src/
  prepare_data.py               filters raw data to one brand, reconstructs threads
  intents.py                    13-intent taxonomy + keyword classifier + TF-IDF/LogReg
  grounding.py                  retrieval-grounded reply drafting
  routing.py                    auto-handle vs escalate decision logic
  pipeline.py                   train/predict-on pipeline (train and eval sets kept separate)
eval/
  golden_set_template.csv       blank template showing the golden-set column format
  golden_set_candidate.csv      stratified sample, pre-manual-review
  golden_set.csv                the real 196-example hand-reviewed golden set
  build_golden_set.py           reproduces the CANDIDATE (stratified sampling + bootstrap
                                 labels) -- not the final file, since manual review isn't
                                 scriptable; see decision_log.md
  eval_harness.py                automated metrics + LLM-as-judge + judge/human agreement
  baselines.py                   trivial + keyword baseline comparisons
report.md                        problem framing, results, failure analysis, the mandatory
                                  "what's misleading" section, next steps
decision_log.md                  13 non-obvious decisions and why
```

## Where the data came from
I don't have direct internet/Kaggle access in my own environment, so the
real ~500MB/3M-row `twcs.csv` was filtered down to AppleSupport-relevant rows
on a Windows machine using PowerShell `Select-String` (see `data/README.md`
for the exact commands), producing a ~39MB file. That was cleaned (encoding
fixes, dropped ~193 rows corrupted by the line-based text filter) down to
~187K clean rows containing 97,927 real AppleSupport agent replies, then
reconstructed into 69,389 real customer↔agent thread pairs.

## What's real vs. what's a known limitation
- **Real**: the data, the 13-intent taxonomy (derived by reading actual
  messages), the golden set (196 examples, manually reviewed), the model,
  the numbers, the baselines, the failure analysis.
- **Known limitation** (see report.md Section 5, the mandatory section):
  the model's training labels are bootstrap-labelled by the keyword
  classifier rather than independently hand-labelled, which is the direct
  cause of the model underperforming both baselines. This is flagged, not
  hidden, and is exactly the kind of finding that section is asking for.
