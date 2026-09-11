"""
Filters the raw Customer-Support-on-Twitter CSV down to one brand and
reconstructs (customer_message, agent_reply) thread pairs.

Usage:
    python src/prepare_data.py --list-brands data/twcs.csv
    python src/prepare_data.py --input data/twcs.csv --brand AmazonHelp --sample 3000 --out data/brand_threads.csv

Also works directly on data/sample_brand_tweets.csv for a quick smoke test:
    python src/prepare_data.py --input data/sample_brand_tweets.csv --brand AmazonHelp --out data/brand_threads.csv
"""
import argparse
import pandas as pd


def read_csv_robust(path: str) -> pd.DataFrame:
    """The real Twitter dataset has some non-UTF-8 bytes (Windows-1252 curly
    quotes etc). Also: if this file was produced by a line-based text filter
    (e.g. PowerShell Select-String on a huge raw file), a small number of rows
    with embedded newlines inside quoted tweet text get split/corrupted --
    skip those with on_bad_lines='skip' rather than failing the whole load."""
    before = sum(1 for _ in open(path, "rb"))
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip", engine="c")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="cp1252", on_bad_lines="skip", engine="c")
    # Drop any surviving garbage rows where 'inbound' isn't literally True/False
    # (remnants of a split multi-line row that still parsed to 7 columns by luck)
    df = df[df["inbound"].isin(["True", "False"])]
    dropped = before - 1 - len(df)  # -1 for header
    if dropped > 0:
        print(f"[prepare_data] Skipped {dropped} malformed/corrupted rows "
              f"(likely multi-line tweets split by a line-based pre-filter)")
    return df


def list_brands(path: str, top_n: int = 20):
    df = read_csv_robust(path)
    agent_rows = df[df["inbound"].astype(str).str.lower() == "false"]
    counts = agent_rows["author_id"].value_counts().head(top_n)
    print("Top brand accounts by reply volume:")
    for name, n in counts.items():
        print(f"  {name}: {n} replies")


def build_thread_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    df = df.copy()
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype(str)

    agent_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]
    customer_msgs = df[df["inbound"]].set_index("tweet_id")

    pairs = []
    for _, reply in agent_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id in customer_msgs.index:
            cust = customer_msgs.loc[parent_id]
            # .loc can return a DataFrame if duplicate ids exist; guard against that
            if isinstance(cust, pd.DataFrame):
                cust = cust.iloc[0]
            pairs.append({
                "thread_id": reply["tweet_id"],
                "customer_tweet_id": parent_id,
                "customer_text": cust["text"],
                "agent_text": reply["text"],
                "created_at": reply.get("created_at", ""),
            })
    return pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to raw CSV")
    ap.add_argument("--list-brands", help="just list top brand accounts and exit")
    ap.add_argument("--brand", help="brand author_id to filter to, e.g. AmazonHelp")
    ap.add_argument("--sample", type=int, default=None, help="cap number of threads (random sample)")
    ap.add_argument("--out", default="data/brand_threads.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.list_brands:
        list_brands(args.list_brands)
        return

    if not args.input or not args.brand:
        raise SystemExit("Provide --input and --brand (or use --list-brands)")

    df = read_csv_robust(args.input)
    pairs = build_thread_pairs(df, args.brand)
    print(f"Found {len(pairs)} customer<->agent thread pairs for brand '{args.brand}'")

    if args.sample and len(pairs) > args.sample:
        pairs = pairs.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {len(pairs)} threads")

    pairs.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
