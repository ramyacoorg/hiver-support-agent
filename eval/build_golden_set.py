"""
Builds a golden-set CANDIDATE file (pre-manual-review) from real thread data.
This is the exact method used to produce eval/golden_set.csv in this repo --
running it reproduces the CANDIDATE, not the final file, because the final
file also required manually reading all 196 rows and hand-correcting ~40
keyword mislabels (see decision_log.md and report.md Section 4/5 -- that step
is inherently not scriptable, it required actually reading the tweets).

Sampling: STRATIFIED, not pure random. "other" alone is ~74% of real
AppleSupport messages (the keyword rules only positively match ~26%), so a
pure random sample of ~200 would make most named intents statistically
invisible. Instead: up to `--per-intent` examples per named intent bucket,
plus `--other-n` from "other", to guarantee every intent has enough support
to compute a meaningful precision/recall.

Intent label: keyword classifier's best guess (see src/intents.py) -- a
labelling AID, not a substitute for review.

Routing label (ideal_routing): derived from what AppleSupport's own agents
ACTUALLY did historically -- if the real agent reply asked the customer to
DM, that means the issue needed account-specific/private handling, so
ideal_routing = "escalate". If the real agent resolved it publicly (gave a
workaround, article link, or asked a public clarifying question without
requesting DM), ideal_routing = "auto_handle". This uses real historical
brand behavior as ground truth rather than a guess.

Usage:
    python eval/build_golden_set.py --threads data/brand_threads_full.csv \
        --out eval/golden_set_candidate.csv --per-intent 12 --other-n 40
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from intents import KeywordIntentClassifier


def build(threads_path: str, out_path: str, per_intent: int, other_n: int, seed: int = 7):
    df = pd.read_csv(threads_path)

    kw = KeywordIntentClassifier()
    df["bootstrap_intent"] = df["customer_text"].apply(lambda t: kw.predict(t).intent)
    df["ideal_routing"] = df["agent_text"].apply(
        lambda a: "escalate" if "dm" in str(a).lower() else "auto_handle"
    )

    parts = []
    for intent, group in df.groupby("bootstrap_intent"):
        n = other_n if intent == "other" else per_intent
        parts.append(group.sample(min(n, len(group)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    sample = sample.rename(columns={"bootstrap_intent": "true_intent"})
    sample["notes"] = "intent=keyword-classifier candidate; routing=matches brand's actual historical action"
    cols = ["thread_id", "customer_tweet_id", "customer_text", "agent_text",
            "created_at", "true_intent", "ideal_routing", "notes"]
    sample = sample[[c for c in cols if c in sample.columns]]

    sample.to_csv(out_path, index=False)
    print(f"Wrote {len(sample)} golden CANDIDATE examples to {out_path}")
    print("This is a candidate for manual review, not a finished golden set --")
    print("see decision_log.md for why the manual pass matters.")
    print(sample["true_intent"].value_counts())
    print(sample["ideal_routing"].value_counts())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/brand_threads_full.csv")
    ap.add_argument("--out", default="eval/golden_set_candidate.csv")
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--other-n", type=int, default=40)
    args = ap.parse_args()
    build(args.threads, args.out, args.per_intent, args.other_n)
