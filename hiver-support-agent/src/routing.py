"""
Auto-handle vs. escalate-to-human decision.

Kept deliberately simple and rule-based so it's fully explainable (a reviewer
can read the rules and know exactly why a message was routed a certain way --
this matters more for a support agent than a marginally higher accuracy score).
Tune the thresholds and the sensitive-keyword list after you look at your
brand's real failure cases.
"""
from dataclasses import dataclass
from typing import List

SENSITIVE_KEYWORDS = [
    "legal", "lawsuit", "attorney", "fraud", "scam", "unauthorized charge",
    "hacked", "data breach", "suicide", "threat", "police",
]

LOW_CONFIDENCE_THRESHOLD = 0.5
NO_RETRIEVAL_MATCH_METHODS = {"fallback_no_match"}


@dataclass
class RoutingDecision:
    action: str  # "auto_handle" | "escalate"
    reason: str


def decide(intent_confidence: float, reply_method: str, message: str) -> RoutingDecision:
    text_lower = message.lower()

    hit = next((kw for kw in SENSITIVE_KEYWORDS if kw in text_lower), None)
    if hit:
        return RoutingDecision(action="escalate",
                                reason=f"Message contains sensitive keyword '{hit}' -- routed to human.")

    if intent_confidence < LOW_CONFIDENCE_THRESHOLD:
        return RoutingDecision(action="escalate",
                                reason=f"Intent classifier confidence {intent_confidence:.2f} "
                                       f"below threshold {LOW_CONFIDENCE_THRESHOLD} -- unclear what "
                                       f"the customer needs.")

    if reply_method in NO_RETRIEVAL_MATCH_METHODS:
        return RoutingDecision(action="escalate",
                                reason="No sufficiently similar past resolved case found to ground a reply.")

    return RoutingDecision(action="auto_handle",
                            reason=f"Intent confidence {intent_confidence:.2f} is high and a similar "
                                   f"past resolution was found -- safe to auto-handle.")
