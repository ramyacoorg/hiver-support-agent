# Data

## What's in this folder

| File | Size | What it is |
|---|---|---|
| `sample_brand_tweets.csv` | 8KB | Fully synthetic, 36 rows. Zero-dependency smoke test for the code -- not used for any real result. |
| `brand_threads_train.csv` | 868KB | Real data. 3,000-thread training subsample used to train the classifier and build the reply-grounding retrieval corpus. **Required for the Quickstart in the main README.** |
| `pipeline_output.csv` | (generated) | Model predictions on the golden set, written by `src/pipeline.py`. |
| `apple_filtered_clean.csv`, `brand_threads_full.csv`, `brand_threads_full_labelled.csv`, `twcs_apple_full.csv` | 20-37MB each | Real data, large intermediates. Git-ignored (see `.gitignore`) -- not needed to reproduce headline results, only to rebuild the golden set from scratch on a different sample. Regenerate them yourself if needed (see below); they're not bundled here to keep the repo light. |

## Where the real data came from

The real "Customer Support on Twitter" dataset (Kaggle,
`thoughtvector/customer-support-on-twitter`, ~500MB/~3M rows) isn't something
I could download directly in my own environment. It was filtered down to
AppleSupport-relevant rows on a Windows machine using PowerShell:

```powershell
$header = Get-Content twcs.csv -TotalCount 1
$header | Set-Content apple_filtered.csv
Select-String -Path twcs.csv -Pattern "AppleSupport" -SimpleMatch | ForEach-Object { $_.Line } | Add-Content apple_filtered.csv
```

That produced a ~39MB file, which was then:
1. Cleaned (fixed a Windows-1252 encoding issue, dropped ~193 rows that a
   line-based text filter corrupted because some tweet text has embedded
   newlines inside quoted CSV fields) → `apple_filtered_clean.csv` (~187K
   clean rows, 97,927 of them real AppleSupport agent replies).
2. Reconstructed into customer↔agent thread pairs by matching each
   AppleSupport reply to the tweet it replied to (`src/prepare_data.py`) →
   `brand_threads_full.csv` (69,389 real thread pairs).
3. Subsampled to 3,000 threads for training/grounding → `brand_threads_train.csv`.
4. Separately, stratified-sampled and manually reviewed to build the
   196-example golden set → `../eval/golden_set.csv` (see `../decision_log.md`).

## Rebuilding from scratch (only needed if you want to change brand/sample size)

```bash
python ../src/prepare_data.py --input apple_filtered_clean.csv --brand AppleSupport --sample 3000 --out brand_threads_train.csv
python ../src/prepare_data.py --input apple_filtered_clean.csv --brand AppleSupport --out brand_threads_full.csv
python ../eval/build_golden_set.py --threads brand_threads_full.csv --out ../eval/golden_set_candidate.csv --per-intent 12 --other-n 40
```
The last command reproduces a stratified **candidate** -- turning it into a
real golden set still requires manually reading and correcting it (that step
is what actually validated the taxonomy; see `../decision_log.md`).

Don't have `apple_filtered_clean.csv`? See the PowerShell command above (or
download `twcs.csv` directly from Kaggle if you have bandwidth for the full
500MB file) to regenerate it.
