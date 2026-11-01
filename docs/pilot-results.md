# Pilot results: off-the-shelf detectors on the frozen corpus (2026-11-01, Rui)

Preliminary. First run of score → calibrate → phase on the real corpus (sha256 8fba77d9, editing round 1), with two
detectors that need no training and run on a laptop. The point was to check the pipeline end to end on real data
and get first numbers while the fine-tuned and 7-8B zero-shot detectors are pending.

## What was run

- **Detectors** (`score_pretrained.py`, used as released): `openai_roberta`, the OpenAI GPT-2 output detector
  (2019), and `chatgpt_roberta`, the HC3 ChatGPT detector (2023). Score = logit(AI) − logit(human), each sentence in
  isolation, on calib and test only (514,005 versions, about 25 minutes per detector on an Apple M4).
- **Calibration**: unchanged `calibrate.py` (per-source conformal p-values, 10 length bins, about 15,000 calib
  sentences per source); no NaN p-values.
- **Phase**: whole mode on test documents with at least 16 sentences (1,301 documents, review item 46) and windows of
  16 and 32; 6 editors (3 models × edit/light); theta in {0.1, ..., 0.9} plus human-only documents; conformal null.
- **Not run**: the Falcon-7B zero-shot scores (too large for 16 GB; a Falcon3-1B pair ran at 2 sentences/s, 66 h),
  ModernBERT, the embedding baseline.

Code: branches `rui/ci-eps` (intervals, ε̂, per-document output, `report.py`), `rui/doc-lengths` (PR #4),
`rui/lm-splits` (`score_pretrained.py`, calib/test-only scoring). Outputs stay local under `data_dir`.

## 1. Per-sentence signal is weak or absent

AUC of edited vs human test sentences (all sources; 0.5 = no signal):

| editor | openai_roberta | chatgpt_roberta |
|---|---|---|
| gemma-4-31b | 0.483 | 0.619 |
| gemma-4-31b-light | 0.509 | 0.610 |
| gpt-oss-120b | 0.399 | 0.540 |
| gpt-oss-120b-light | 0.423 | 0.546 |
| qwen3.5-122b | 0.460 | 0.609 |
| qwen3.5-122b-light | 0.483 | 0.609 |

- `openai_roberta` has no signal; it scores edited sentences as slightly *more* human (gpt-oss most).
- `chatgpt_roberta` has a real but weak signal: the median edited z (`mu_eff`) is 0.6-0.9 on ELLIPSE, 0.2-0.5 on
  PERSUADE, Enron, FineWeb-Edu and OpenReview, and about 0 on peS2o, WritingPrompts and XSum.
- gpt-oss edits are the hardest to detect for both. Light and full edits score about the same, although light
  edits change about half as many words (median 25% vs 50%).

## 2. HC detects only where the theory says a weak signal can be detected

![phase diagram, chatgpt_roberta](figures/pilot/phase_chatgpt_roberta.png)

Each edited test document (one draw per editor and theta) is placed at its own theta = 1 − log m / log n and
r = max(mean edited z, 0)² / (2 log n), and binned; colour is the share HC rejects, with thresholds that account for
within-document correlation (section 3). r uses the true edit positions
and the same z the test sees, so this placement is descriptive, not an estimate. Most mass sits at r < 0.5, below
the almost-full-recovery line, and 30% of draws at r = 0.

![HC power by theta](figures/pilot/power.png)

With the correlated thresholds of section 3, HC on `chatgpt_roberta` rejects 11.1% of edited documents at
theta = 0.1 (dense), 6.4% at 0.5 and 4.9% at 0.9, against 4.7% on human-only documents: detectable when dense,
indistinguishable from the null when sparse. (With the old iid thresholds: 17.3%, 11.1%, 9.2% against 8.9%.)
Short documents are now detected more often than long ones (12.6% for n = 16-31 vs 6.8% for n ≥ 64 at
theta = 0.1): with a shared document component, extra sentences add little independent information (effective
sample size n / (1 + (n − 1) rho) levels off at about 1 / rho), while the threshold keeps growing with n. A shift of
z ≈ 0.4 at n = 16 is r ≈ 0.03, far below the detection boundary for theta > 1/2 (r = 0.2 at theta = 0.7), so this
is the expected behaviour of a weak detector, not a failure of HC. Support recovery fails everywhere (BH exact
recovery 0%, Hamming error / m ≈ 1). `openai_roberta` stays at or below its null rate at every theta.

[phase diagram, openai_roberta](figures/pilot/phase_openai_roberta.png)

## 3. Within-document dependence breaks the iid thresholds; correlated thresholds restore the level

![type I error](figures/pilot/type1.png)

Rejection rate on human-only documents (whole, n ≥ 16, 1,301 documents, nominal 0.05), with the iid thresholds
used so far and with thresholds for the within-document correlation:

| test | openai_roberta, iid | openai_roberta, correlated | chatgpt_roberta, iid | chatgpt_roberta, correlated |
|---|---|---|---|---|
| min-p | 0.045 | 0.046 | 0.057 | 0.060 |
| HC | 0.059 | 0.051 | 0.089 | 0.047 |
| HC+ | 0.111 | 0.055 | 0.172 | 0.046 |
| Stouffer | 0.145 | 0.050 | 0.238 | 0.038 |

Windows of 16 and 32 give the same picture (correlated: 0.03-0.07 for every test and detector).

With synthetic i.i.d. scores on the same test split (whole documents, before the length cutoff) Stouffer is 0.046, so the inflation comes from the text. The
intraclass correlation of human sentence z within test documents is 0.07-0.28 per source for `chatgpt_roberta`
(ELLIPSE 0.28, PERSUADE 0.23) and 0.02-0.09 for `openai_roberta`: a document's sentences share author and topic, so
their scores move together. Tests that pool many moderate p-values (Stouffer, HC+) suffer most; min-p, which uses
only the smallest, holds its level. The original Monte Carlo thresholds assumed independence (review items 41, 44).

**Fix** (review item 41):
1. `calibrate.py` cross-fits the calib split: calib documents are split in two folds (by a hash of seed and doc_id,
   the same for every detector), and each fold's sentences get p-values against the other fold
   (`pvalues_calib/<detector>.parquet`). Test p-values are unchanged (checked byte for byte).
2. `phase.py` estimates one correlation rho per source and detector from those calib documents (one-way ANOVA
   intraclass correlation). For `chatgpt_roberta`: ELLIPSE 0.26, PERSUADE 0.23, FineWeb-Edu 0.17, Enron 0.16,
   XSum 0.16, WritingPrompts 0.12, peS2o 0.12, OpenReview 0.09, within 0.02 of the same estimate on test documents.
3. Thresholds by Monte Carlo under the equicorrelated model z_j = sqrt(rho) a + sqrt(1 − rho) e_j
   (`stats.null_thresholds(..., rho)`), per source and document length, as a third null type,
   `conformal_dependent`. Stouffer's threshold, for example, becomes 1.645 sqrt(1 + (n − 1) rho): 3.66 instead of
   1.64 for ELLIPSE at n = 16. A run takes about 14 minutes instead of 3.
4. Not done: a richer Sigma (Toeplitz, `hc_variants`), or PC-adjusted / innovated HC (Lecture 4) to remove the
   shared component before testing, which suits sparse cells.

## 4. Edited fraction

With these detectors the estimators carry little information. Storey's estimate stays at 0.16-0.24 for
`chatgpt_roberta` whatever the true fraction (0.04-0.66); the Meinshausen-Rice lower bound is about 0.01 with iid
thresholds and 0.003 with correlated ones, covering in 98% and 99.6% of draws.

## Next

1. The fine-tuned ModernBERT (Ege) through the same pipeline: the run most likely to place documents in the
   detectable region. The 7-8B zero-shot scores need a GPU (RunPod).
2. Check the equicorrelated model against a direct estimate (95th percentile of each test on calib documents), and
   compare with PC-adjusted HC.
3. ε̂ per cell (all documents pooled) instead of per document.
4. Comparisons for the report: detector strength vs detectability, held-out editor (Qwen), sources, light vs full.

Reproduce (with `data_dir` pointing at a built `versions.parquet`):

```bash
uv run python -m sparse_edits.score_pretrained
uv run python -m sparse_edits.calibrate
uv run python -m sparse_edits.phase --what simulate
uv run python -m sparse_edits.phase --what empirical
uv run python -m sparse_edits.report
```
