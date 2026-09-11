"""
Evaluation harness.

Two parts, per the assignment:
1. Automated metrics: intent classification accuracy/F1 and routing accuracy
   against your hand-labelled golden set.
2. LLM-as-judge rubric for reply QUALITY (there's no single "correct" reply
   text, so this can't be exact-match scored) -- PLUS a check of how well the
   judge agrees with a human, which the assignment explicitly requires.

How to use:
1. Fill in eval/golden_set.csv (copy golden_set_template.csv, hand-label
   150-250 rows sampled from your brand_threads.csv -- see the README for a
   sampling strategy).
2. Run: python eval/eval_harness.py --golden eval/golden_set.csv --predictions data/pipeline_output.csv
3. For the judge-agreement check: this script also asks YOU (interactively, or
   from a pre-filled human_score column) to score a subset of replies 1-5, then
   reports how often the LLM judge and your score agree (exact + within-1).
"""
import argparse
import json
import os
import urllib.request

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def classification_metrics(golden: pd.DataFrame, preds: pd.DataFrame) -> dict:
    merged = golden.merge(preds, on="thread_id", suffixes=("_true", "_pred"))
    y_true = merged["true_intent"]
    y_pred = merged["predicted_intent"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("Intent classification report:\n", classification_report(y_true, y_pred, zero_division=0))

    routing_acc = accuracy_score(merged["ideal_routing"], merged["routing_action"])

    return {"intent_accuracy": acc, "intent_macro_f1": f1, "routing_accuracy": routing_acc,
            "n_examples": len(merged)}


def judge_reply_quality(customer_text: str, drafted_reply: str, api_key: str) -> dict:
    """LLM-as-judge: rates the drafted reply 1-5 on a rubric, with justification."""
    prompt = (
        "Rate this customer support reply on a 1-5 scale for: "
        "(a) relevance to the customer's issue, (b) whether it takes a concrete "
        "next step, (c) tone. Respond ONLY with JSON: "
        '{"score": <1-5 int>, "justification": "<one sentence>"}\n\n'
        f"Customer message: {customer_text}\n"
        f"Drafted reply: {drafted_reply}"
    )
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    text = "".join(b.get("text", "") for b in data.get("content", []))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "justification": f"unparsed: {text}"}


def judge_agreement_check(golden: pd.DataFrame, preds: pd.DataFrame, api_key: str, n: int = 15):
    """
    Runs the LLM judge on a subset AND expects a 'human_score' column in the
    golden set (you fill this in by hand for the same subset) so you can report
    judge<->human agreement -- required by the assignment.
    """
    merged = golden.merge(preds, on="thread_id", suffixes=("_true", "_pred")).head(n)
    if "human_score" not in merged.columns:
        print("\n[!] No 'human_score' column found in golden set. Add one, hand-score "
              "these same rows 1-5 yourself, and re-run this function to get the "
              "judge<->human agreement number the report requires.")
        return None

    judge_scores, human_scores = [], []
    for _, row in merged.iterrows():
        result = judge_reply_quality(row["customer_text"], row["drafted_reply"], api_key)
        judge_scores.append(result.get("score"))
        human_scores.append(row["human_score"])

    pairs = [(j, h) for j, h in zip(judge_scores, human_scores) if j is not None]
    exact = sum(1 for j, h in pairs if j == h) / len(pairs)
    within_1 = sum(1 for j, h in pairs if abs(j - h) <= 1) / len(pairs)
    print(f"\nJudge<->human agreement over {len(pairs)} examples: "
          f"exact={exact:.2%}, within_1={within_1:.2%}")
    return {"exact_agreement": exact, "within_1_agreement": within_1, "n": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.csv")
    ap.add_argument("--predictions", default="data/pipeline_output.csv")
    ap.add_argument("--judge-n", type=int, default=15)
    args = ap.parse_args()

    if not os.path.exists(args.golden):
        raise SystemExit(f"{args.golden} not found -- fill in eval/golden_set_template.csv first")

    golden = pd.read_csv(args.golden)
    preds = pd.read_csv(args.predictions)

    metrics = classification_metrics(golden, preds)
    print("\n=== Automated metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        judge_agreement_check(golden, preds, api_key, n=args.judge_n)
    else:
        print("\n[i] Set ANTHROPIC_API_KEY to also run the LLM-as-judge reply-quality "
              "scoring + judge<->human agreement check.")


if __name__ == "__main__":
    main()
