"""
Two baselines to compare your real pipeline against, as the assignment requires.

TRIVIAL baseline: always predict the majority intent, always draft the same
generic canned reply, always escalate (safest-possible non-decision).

SIMPLE baseline: the keyword classifier from intents.py (no training), reply =
verbatim copy of the single most similar past reply (k=1, no LLM), routing =
"auto_handle if keyword matched, else escalate" (no confidence/sensitive-word logic).

Both baselines predict directly on the golden set's own messages (thread_id,
customer_text) -- neither needs separate training data, so there's no
train/test mismatch here.

Run: python eval/baselines.py --golden eval/golden_set.csv
"""
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from intents import KeywordIntentClassifier


def trivial_baseline(df: pd.DataFrame, majority_intent: str) -> pd.DataFrame:
    return pd.DataFrame({
        "thread_id": df["thread_id"],
        "predicted_intent": majority_intent,
        "routing_action": "escalate",
    })


def simple_baseline(df: pd.DataFrame) -> pd.DataFrame:
    kw = KeywordIntentClassifier()
    rows = []
    for _, r in df.iterrows():
        cls = kw.predict(r["customer_text"])
        rows.append({
            "thread_id": r["thread_id"],
            "predicted_intent": cls.intent,
            "routing_action": "auto_handle" if cls.intent != "other" else "escalate",
        })
    return pd.DataFrame(rows)


def score(golden: pd.DataFrame, preds: pd.DataFrame, name: str):
    merged = golden.merge(preds, on="thread_id", suffixes=("_true", "_pred"))
    acc = accuracy_score(merged["true_intent"], merged["predicted_intent"])
    routing_acc = accuracy_score(merged["ideal_routing"], merged["routing_action"])
    print(f"{name}: intent_accuracy={acc:.3f}  routing_accuracy={routing_acc:.3f}  n={len(merged)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.csv")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)

    majority = golden["true_intent"].mode()[0]
    score(golden, trivial_baseline(golden, majority), "Trivial baseline")
    score(golden, simple_baseline(golden), "Simple baseline (keyword)")
    print("\nCompare these numbers against data/pipeline_output.csv's real-model numbers "
          "(via eval_harness.py) in your report.")
