# Report — AppleSupport AI Support Agent

## 1. Problem framing
Looking at AppleSupport's real historical replies across ~69K threads, "good"
for this brand does NOT mean resolving every issue publicly in-thread —
roughly half of real replies ask the customer to DM, and many of the rest ask
a clarifying question rather than giving a final answer. So "good" here means:
correctly recognize *which* messages are simple enough to answer with a public
link/workaround (auto-handle) versus which genuinely need a human follow-up
(escalate), and correctly bucket the underlying issue so a human agent (or a
future richer system) doesn't have to re-triage from scratch.

I deliberately did NOT build: full multi-turn DM-thread handling (out of scope
-- the dataset only has the public thread), an actual troubleshooting
knowledge base beyond what's inferable from past tweets, or account/order
lookups (no such system exists to connect to here).

## 2. Approach
- **Data**: real Kaggle "Customer Support on Twitter" data. I received it
  pre-filtered to AppleSupport via a PowerShell text search on the full
  ~500MB/3M-row file (I don't have direct access to Kaggle or the raw file
  myself), yielding ~205K rows -> **69,389 real customer<->agent thread
  pairs** after reconstruction (`src/prepare_data.py`).
- **Intent taxonomy**: derived by reading a broad random sample of real
  customer messages (not guessed), landing on 13 intents plus "other" --
  including a distinct `autocorrect_i_bug` bucket for a real, very
  high-frequency iOS 11 bug (autocorrect turning "I" into a glitch symbol).
  See `decision_log.md` for the full list and reasoning.
- **Golden evaluation set**: 196 examples, **stratified** (not purely random)
  -- up to 12 per intent bucket plus 40 from "other" -- sampled from the full
  69,389 threads, specifically so minority intents aren't invisible in a
  196-row sample of a corpus where "other" alone is ~74% of messages. I then
  manually read every single one of the 196 (see `decision_log.md`) and
  corrected ~40 keyword-classifier mislabels, plus dropped and replaced 2
  rows corrupted by the PowerShell line-based pre-filter. `ideal_routing` was
  set programmatically from the brand's own real historical reply pattern
  (asks to DM / asks a question -> escalate; gives a direct link/workaround
  with neither -> auto_handle) -- grounded in actual brand behavior, not
  invented.
- **Classification**: `src/intents.py` -- keyword baseline (rules, no
  training) and TF-IDF + Logistic Regression, trained on a **3,000-thread
  pool disjoint from the golden set** (bootstrap-labelled by the keyword
  classifier, since no larger independently-labelled set exists).
- **Reply drafting**: `src/grounding.py` -- TF-IDF retrieval over that same
  3,000-thread training pool (also disjoint from the golden set, so a golden
  example can never retrieve itself as its own historical precedent), reply
  adapted from the closest real historical reply.
- **Routing**: `src/routing.py` -- rule-based on intent confidence, retrieval
  match, and a sensitive-keyword list.
- **Evaluation methodology**: `src/pipeline.py` now supports separate
  `--train` and `--predict-on` sets specifically so headline numbers reflect
  prediction on genuinely unseen examples, not the training data.

## 3. Results vs. baselines
All three rows below predict on the exact same 196 real golden-set messages.

| Metric | Trivial baseline | Simple baseline (keyword) | My model (TF-IDF+LogReg) |
|---|---|---|---|
| Intent accuracy | 0.138 | **0.776** | 0.265 |
| Intent macro-F1 | — | — | 0.236 |
| Routing accuracy | **0.745** | 0.398 | 0.337 |

Judge-rated reply quality: not run -- no `ANTHROPIC_API_KEY` configured in
this environment. `eval/eval_harness.py` supports it and would run
`judge_agreement_check()` given a key and a `human_score` column.

**The trained model loses to both baselines. This is the real result, not a
bug I'm hiding** -- see Section 5 for why, and why the "winning" baseline
numbers are themselves not something to celebrate uncritically either.

## 4. Failure analysis — top 5 failure modes
1. **The trained TF-IDF+LogReg model collapsed to predicting "other" for 164
   of 196 messages (84%).** Its bootstrap training labels (from the keyword
   classifier, run over the 3,000-thread training pool) are themselves ~74%
   "other", because the keyword rules only positively match ~26% of real
   messages. A model trained on that label distribution learns "when in
   doubt, say other" -- and with messy short tweets, it's almost always in
   doubt. *Hypothesis: weak/bootstrap supervision from low-coverage keyword
   rules poisons the very model meant to generalize past those rules.*
2. **"Other" has recall=1.00 but precision=0.14** in the classification
   report -- the model isn't actually good at recognizing genuine "other"
   messages, it's just predicting "other" so often that it can't miss one.
   Real example: thread with "how do I open PDF pages in keynote as a
   slideshow" was correctly identifiable as `how_to_question` by a human in
   two seconds, but the model said "other". *Hypothesis: this metric alone
   would look fine in isolation ("100% recall!") while hiding that the model
   has essentially stopped discriminating.*
3. **Routing accuracy: the trivial always-escalate baseline (0.745) beats my
   real model (0.337)** -- a textbook accuracy paradox. The golden set is
   74.5% "escalate" by the brand's real historical behavior, so a baseline
   that always says "escalate" scores well on accuracy alone without making
   a single real decision. *Hypothesis: routing accuracy on an imbalanced
   label set rewards a non-decision; a metric like balanced accuracy or a
   cost-weighted score (false auto-handle >> false escalate) would be far
   more honest, and is what I'd switch to with another week.*
4. **Keyword-classifier false negatives, found only by manually reading all
   196 golden examples.** Roughly 40 of 196 (~20%) needed relabelling because
   the rules missed real matches on phrasing variants: "Battery Issues"
   didn't match any of my "battery drain" keyword phrases, "Air drop" (two
   words) didn't match my "airdrop" keyword, "keeps self restarting" didn't
   match "keeps restarting". *Hypothesis: exact-substring keyword rules are
   brittle to real-world phrasing variance in a way that isn't visible until
   you actually read the "other" bucket by hand -- automated coverage
   numbers alone would have hidden this.*
5. **Two golden-set candidate rows were corrupted data artifacts, not real
   messages** -- leaked fragments of adjacent rows caused by the line-based
   PowerShell pre-filtering step used to shrink the original 500MB file
   before I could access it. Caught by manual read-through, not by any
   automated check. *Hypothesis: any text-line-based pre-filter on a CSV
   with multi-line quoted fields will silently corrupt a small fraction of
   rows (~8% of the pre-filtered file failed to parse at all, and at least 2
   of the ~205K "successfully parsed" rows were still corrupted); a
   proper CSV-aware filter (not `Select-String`) would avoid this entirely.*

## 5. What is misleading about my headline number? (mandatory)
The most important thing to say clearly: **my real model's headline
accuracy (26.5%) is worse than a two-line keyword lookup (77.6%), and my real
model's headline routing accuracy (33.7%) is worse than a baseline that never
makes a real decision (74.5%).** A superficial read of just the top-line
numbers would suggest I built something actively harmful. That's not quite
right either, for a few reasons worth being upfront about:

- The keyword baseline's 77.6% is **partly circular**: my manual corrections
  to the golden set were made by reading the messages and mentally applying
  something very close to the same keyword logic (I invented the taxonomy by
  reading real messages, then wrote keyword rules for exactly the phrases I'd
  seen). A truly independent labeller using Apple's own internal category
  names might score the keyword baseline much lower.
- The trivial baseline's 74.5% "routing accuracy" reflects zero actual
  judgment -- it's a property of the golden set's class imbalance (which
  itself reflects a heuristic I wrote, not ground truth from Apple), not a
  real capability. Reporting it next to my model's score without this caveat
  would be actively misleading.
- My model's poor accuracy is a *specific, fixable* failure (bootstrap-label
  class imbalance from low keyword coverage), not evidence that "TF-IDF +
  LogReg can't work here" -- with better bootstrap labels (or a real second
  round of hand-labelling for training, not just eval), the same architecture
  would likely improve substantially. I don't have evidence for that claim
  yet, which is exactly why it's a claim and not a result.
- `ideal_routing` in the golden set is *my* operationalization of "how the
  brand historically resolved this" (DM-mention / question -> escalate,
  else auto_handle), not a ground truth Hiver or Apple provided. A different,
  equally defensible operationalization could shift every routing number in
  this report.
- This is still one brand out of dozens in the full dataset, and a 3,000-
  thread training subsample of 69,389 available threads -- a different
  brand or a full-scale run could look very different.

## 6. What I'd do next with one more week
- Fix the bootstrap-label class imbalance directly: either hand-label a
  larger, separate training-label set (not just the 196-example eval set) or
  weight/upsample minority classes before training the TF-IDF+LogReg model.
- Replace routing accuracy with a cost-weighted metric that penalizes a false
  auto-handle far more than a false escalate, so the trivial baseline stops
  looking artificially strong.
- Get an independent second labeller for at least a subset of the golden set
  to check how much of the keyword baseline's strong score is circularity
  versus real taxonomy quality.
- Wire up the LLM-grounded reply mode and LLM-as-judge with a real human
  agreement check (currently untested -- no API key available in this pass).
- Fix the two corrupted-row and ~40 false-negative issues at the *source* --
  a CSV-aware pre-filter instead of line-based text search, and a keyword
  list that handles common phrasing/spelling variants -- rather than
  catching them by hand each time.

## Judge <-> human agreement
Not computed in this pass -- requires `ANTHROPIC_API_KEY` and a hand-scored
subset, neither of which were available here. `eval/eval_harness.py`
implements this and is ready to run once a key is set.
