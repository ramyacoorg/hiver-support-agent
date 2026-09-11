"""
Grounded reply drafting.

Approach: retrieval-augmented, not free-form generation.
1. Build a TF-IDF index over all historical customer_text in the brand's threads.
2. For a new incoming message, retrieve the k most similar past customer messages.
3. Look at how the brand actually replied to those (agent_text) -- this is the
   "history of how the brand resolved similar issues" the assignment asks for.
4. Draft the reply either by (a) adapting the closest historical reply as a
   template, or (b) if you set an ANTHROPIC_API_KEY env var, asking an LLM to
   synthesize a reply *grounded in* the retrieved examples (the LLM is told to
   only use facts/patterns present in them, not invent policy).

Option (b) is commented/optional so the pipeline stays runnable in <15 min with
zero API cost -- exactly what the README promises reviewers.
"""
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GroundedReply:
    reply_text: str
    grounded_on: List[str]  # thread_ids used as evidence
    method: str  # "retrieval_template" or "llm_grounded"


class ReplyGrounder:
    def __init__(self, threads_df: pd.DataFrame, k: int = 3):
        """threads_df needs columns: thread_id, customer_text, agent_text"""
        self.df = threads_df.reset_index(drop=True)
        self.k = k
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text"].fillna(""))

    def retrieve(self, message: str) -> pd.DataFrame:
        q = self.vectorizer.transform([message])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][: self.k]
        result = self.df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def draft_reply(self, message: str) -> GroundedReply:
        retrieved = self.retrieve(message)
        if retrieved.empty or retrieved["similarity"].max() < 0.05:
            return GroundedReply(
                reply_text="Thanks for reaching out -- could you share a bit more detail "
                            "(e.g. order number) so we can look into this for you?",
                grounded_on=[],
                method="fallback_no_match",
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return self._llm_grounded_reply(message, retrieved, api_key)
        return self._template_reply(message, retrieved)

    def _template_reply(self, message: str, retrieved: pd.DataFrame) -> GroundedReply:
        best = retrieved.iloc[0]
        # Adapt the closest historical resolution as-is; this is intentionally
        # simple/explainable -- flag it in your report as a limitation vs. an LLM draft.
        return GroundedReply(
            reply_text=best["agent_text"],
            grounded_on=[str(t) for t in retrieved["thread_id"].tolist()],
            method="retrieval_template",
        )

    def _llm_grounded_reply(self, message: str, retrieved: pd.DataFrame, api_key: str) -> GroundedReply:
        import json
        import urllib.request

        examples = "\n".join(
            f"- Customer: {r.customer_text}\n  Brand replied: {r.agent_text}"
            for _, r in retrieved.iterrows()
        )
        prompt = (
            "You are drafting a customer support reply. Base your reply ONLY on the "
            "patterns shown in these past resolved cases -- do not invent policy details "
            "not present below.\n\n"
            f"Past similar cases:\n{examples}\n\n"
            f"New customer message: {message}\n\n"
            "Write a short, on-brand reply (2-3 sentences):"
        )
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={"Content-Type": "application/json", "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return GroundedReply(
            reply_text=text.strip(),
            grounded_on=[str(t) for t in retrieved["thread_id"].tolist()],
            method="llm_grounded",
        )
