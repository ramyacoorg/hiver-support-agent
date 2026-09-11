"""
End-to-end pipeline: for each incoming customer message ->
  1. classify intent
  2. draft a grounded reply
  3. decide auto-handle vs escalate

Two usage modes:

1. PRODUCTION / cold-start (no golden set yet) -- train and predict on the
   same batch, bootstrapping intent labels from the keyword classifier:
       python src/pipeline.py --threads data/brand_threads.csv --out data/pipeline_output.csv

2. EVALUATION (what actually produces this submission's headline numbers) --
   train the classifier AND build the retrieval-grounding corpus from a large
   separate pool of historical threads, then predict on the real golden-set
   messages. This avoids the circularity of training and testing on the exact
   same examples:
       python src/pipeline.py --train data/brand_threads_train.csv \
           --predict-on eval/golden_set.csv --out data/pipeline_output.csv
"""
import argparse

import pandas as pd

from intents import KeywordIntentClassifier, TfidfIntentClassifier
from grounding import ReplyGrounder
from routing import decide


def run(train_path: str, predict_path: str, out_path: str):
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)

    # Bootstrap intent labels on the TRAINING pool only, using the keyword
    # classifier as a stand-in for a golden set at this stage. This is a real
    # methodology limitation (see report.md Section 5) -- the training labels
    # are not independently verified -- but the training pool is disjoint from
    # the golden set being predicted on below, so at minimum train/test are
    # not the same examples.
    kw_clf = KeywordIntentClassifier()
    train_df["bootstrap_intent"] = train_df["customer_text"].apply(lambda t: kw_clf.predict(t).intent)

    tfidf_clf = TfidfIntentClassifier()
    tfidf_clf.fit(train_df["customer_text"].tolist(), train_df["bootstrap_intent"].tolist())

    # Retrieval corpus for grounded replies is also the training pool only --
    # a golden-set example can never retrieve itself as its own "historical
    # precedent", which would trivially inflate grounding quality.
    grounder = ReplyGrounder(train_df, k=3)

    results = []
    for _, row in predict_df.iterrows():
        msg = row["customer_text"]

        cls = tfidf_clf.predict(msg)
        grounded = grounder.draft_reply(msg)
        routing = decide(cls.confidence, grounded.method, msg)

        results.append({
            "thread_id": row["thread_id"],
            "customer_text": msg,
            "predicted_intent": cls.intent,
            "intent_confidence": round(cls.confidence, 3),
            "drafted_reply": grounded.reply_text,
            "grounded_on_threads": ";".join(grounded.grounded_on),
            "reply_method": grounded.method,
            "routing_action": routing.action,
            "routing_reason": routing.reason,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Trained on {len(train_df)} threads, predicted on {len(out_df)} messages -> {out_path}")
    print("\nRouting breakdown:")
    print(out_df["routing_action"].value_counts())
    print("\nSample output:")
    print(out_df[["customer_text", "predicted_intent", "routing_action"]].head(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", help="shortcut: use this file as both --train and --predict-on")
    ap.add_argument("--train", help="training pool for the classifier + grounding corpus")
    ap.add_argument("--predict-on", help="file to run predictions on (needs thread_id, customer_text)")
    ap.add_argument("--out", default="data/pipeline_output.csv")
    args = ap.parse_args()

    train_path = args.train or args.threads or "data/brand_threads_train.csv"
    predict_path = args.predict_on or args.threads or train_path
    run(train_path, predict_path, args.out)
