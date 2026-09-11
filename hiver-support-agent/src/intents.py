"""
Intent classification for customer support messages.

Two modes, both included so your report can compare them (baseline vs real model):

1. `KeywordIntentClassifier`  -- trivial baseline. Rule/keyword lookup, no training.
2. `TfidfIntentClassifier`    -- simple baseline. TF-IDF + Logistic Regression,
   trained on whatever labelled examples you give it (e.g. your golden set).

Intents are NOT hardcoded here on purpose -- you derive them by reading a sample
of your brand's real messages (see notebook/skim step in the README) and pass
your taxonomy in. Below is an EXAMPLE taxonomy for the sample AmazonHelp data;
replace with your own once you've looked at your real brand's data.
"""
from dataclasses import dataclass
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Real taxonomy derived from reading a broad random sample of the ~69K real
# AppleSupport threads (the full uploaded twcs.csv, filtered to AppleSupport).
# Every category below was found by actually reading real customer messages,
# not guessed upfront -- see decision_log.md for the walkthrough. Dict order
# matters: more specific intents are checked before general ones.
EXAMPLE_INTENT_KEYWORDS: Dict[str, List[str]] = {
    # A famous, very high-frequency real iOS 11 bug: autocorrect kept turning
    # "I" into a question-mark symbol. Distinctive and common enough in the
    # real data to deserve its own bucket rather than "general_complaint".
    "autocorrect_i_bug": ["i? i?", "capital \u201ci\u201d", "the word i? like",
                           "changed to this", "i.t.\u201d", "i? need my i? back"],
    "battery_drain": ["battery draining", "battery drain", "battery life", "battery is worse",
                       "battery worse", "drains it down", "drains battery", "% battery",
                       "battery runs out", "turns off after", "dies on"],
    "performance_or_crash": ["freezes", "freezing", "apps are broken", "apps crash",
                              "apps stop working", "won't open", "not opening",
                              "unresponsive", "hangs", "hanging", "keeps restarting", "lags"],
    "connectivity_issue": ["wifi", "wi-fi", "bluetooth", "captive", "can't connect",
                            "connecting to the server", "airport"],
    "sync_or_cloud_issue": ["icloud", "itunes", "sync", "syncing", "duplicate", "backup",
                             "airdrop", "photos aren't loading", "photos from icloud"],
    "how_to_question": ["how can i", "how do i", "where can i find", "is it okay to",
                         "is it ok to", "wondering if", "any reason"],
    "account_or_purchase": ["applecare", "eligible", "password change", "sign in",
                             "log in", "login", "i-store", "itunes login", "purchase",
                             "warranty", "store team", "banking information",
                             "cloud storage is full", "subscribing"],
    "security_or_login_issue": ["fingerprint", "touch id", "password are incorrect",
                                 "password incorrect", "is this a real", "phishing",
                                 "suspicious", "hacked", "unauthorized"],
    "call_or_network_issue": ["can't make", "cant make", "phone calls", "cellular",
                               "lte", "mobile data", "no service", "no signal", "texting"],
    "notification_issue": ["notification", "notifications", "missed calls", "badge"],
    "feature_request_or_missing": ["missing auto", "bring back", "should consider adding",
                                    "why doesnt", "why doesn't apple", "should add"],
    "positive_or_resolved": ["thank you", "thanks for the response", "fixed it",
                              "sorted it", "going through settings fixed", "appreciate"],
    "general_update_complaint": ["fix this update", "update sux", "ruined my phone",
                                  "ruined the phone", "ruined my", "hate it", "horrible",
                                  "paralysed my phone", "#infuriating", "not happy",
                                  "#nothappy", "when will you fix", "actual fix", "irritating",
                                  "trash", "blows", "worst update", "tired of", "give up on",
                                  "wtf", "fuck", "gonna fix"],
}


@dataclass
class ClassificationResult:
    intent: str
    confidence: float  # 0-1, used later for the auto-handle/escalate decision


class KeywordIntentClassifier:
    """Trivial baseline: first keyword match wins. No training needed."""

    def __init__(self, keywords: Dict[str, List[str]] = None):
        self.keywords = keywords or EXAMPLE_INTENT_KEYWORDS

    def predict(self, text: str) -> ClassificationResult:
        t = text.lower()
        for intent, kws in self.keywords.items():
            for kw in kws:
                if kw in t:
                    return ClassificationResult(intent=intent, confidence=0.6)
        return ClassificationResult(intent="other", confidence=0.3)


class TfidfIntentClassifier:
    """Simple baseline: TF-IDF + Logistic Regression, trained on labelled examples."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, text: str) -> ClassificationResult:
        if not self._fitted:
            raise RuntimeError("Call .fit() with labelled examples first")
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        idx = probs.argmax()
        return ClassificationResult(intent=self.clf.classes_[idx], confidence=float(probs[idx]))
