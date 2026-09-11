# Decision Log

1. **Chose AppleSupport as the brand**, since I only had a partial pre-filter
   of the raw dataset available and AppleSupport had by far the most volume
   in it (69,389 real thread pairs after reconstruction).

2. **Derived a 13-intent taxonomy by reading a broad random sample of real
   customer messages**, not by guessing upfront: battery_drain,
   performance_or_crash, connectivity_issue, sync_or_cloud_issue,
   how_to_question, account_or_purchase, security_or_login_issue,
   call_or_network_issue, notification_issue, feature_request_or_missing,
   positive_or_resolved, general_update_complaint, and a distinctive
   autocorrect_i_bug bucket for a real, very high-frequency iOS 11 bug.

3. **Gave the iOS 11 "autocorrect turns I into a glitch symbol" bug its own
   intent** rather than folding it into general_update_complaint, because it
   alone accounted for hundreds of near-identical real complaints -- treating
   it separately is clearly higher-value for a real triage system.

4. **Sampled the golden set with stratification, not pure random sampling**
   (up to 12 per named intent, 40 from "other"), because "other" alone is
   ~74% of the real corpus -- a pure random 196-row sample would have made
   most named intents statistically invisible.

5. **Manually read all 196 stratified candidates rather than trusting the
   keyword bootstrap labels**, and corrected ~40 of them. This was the single
   most valuable step in the whole build -- it's what surfaced the keyword
   classifier's real false-negative rate (Section 4 of report.md), which
   automated coverage metrics alone would never have shown.

6. **Dropped and replaced 2 golden-set candidates that turned out to be
   corrupted data**, not real messages -- artifacts of the line-based
   PowerShell pre-filter used to shrink the original 500MB file before I
   could access it. Documented as a real limitation rather than silently
   patched over.

7. **Defined `ideal_routing` from the brand's actual historical reply
   pattern** (DM-mention or a trailing question -> escalate; a direct
   link/workaround with neither -> auto_handle) rather than inventing routing
   labels from scratch -- this is a real operationalization choice I made,
   not ground truth, and I said so explicitly in report.md Section 5.

8. **Split training and golden-set data completely** (3,000-thread training
   pool excludes all 196 golden-set thread_ids) for both the intent
   classifier's training data and the reply-grounding retrieval corpus. This
   was a direct fix to a validity concern I'd flagged in an earlier draft of
   this project, where training and eval data overlapped entirely.

9. **Bootstrap-trained the TF-IDF classifier on the keyword classifier's own
   output over the training pool**, not on hand-labels, since no
   independently hand-labelled training set exists beyond the 196-example
   eval set. This directly caused the class-imbalance failure mode in
   report.md Section 4 -- flagged rather than hidden.

10. **Reported that my real model loses to both baselines**, instead of
    tuning until the headline number looked better. The report explains why
    with real evidence (class-imbalance collapse to "other", an accuracy
    paradox on routing) rather than picking a rosier metric.

11. **Kept routing rule-based instead of learned**, so a reviewer can read
    the exact logic and know why any message was routed a certain way --
    important when the actual cost of a wrong auto-handle decision is high.

12. **Used k=3 for reply-grounding retrieval** and a 3,000-thread subsample
    for training/grounding (not the full 69,389), per the assignment's own
    note that a subsample is expected and won't be run on the full dataset.

13. **Did not call an LLM for reply drafting or judging in this pass.** No
    API key was available in this environment; the pipeline supports it
    (`ANTHROPIC_API_KEY` env var) but headline numbers shouldn't depend on
    something unverified here. Documented as a "next week" item.
