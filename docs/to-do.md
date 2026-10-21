# To do

Everything implemented so far is preliminary. The C4-only version ran end to end on a smoke test (48 C4 documents,
a local gemma3:1b editor, tiny stand-in detector models); the multi-source corpus step has been run on its own. No
setting is final. Open decisions are in
[review.md](review.md).

## Implemented (preliminary)

- [x] Corpus: eight human-text sources (C4, PERSUADE, ELLIPSE, XSum, arXiv abstracts, OpenReview reviews, Enron
      emails, PG-19), sentence segmentation, eligibility filters, document-level splits per source (`sources.py`,
      `corpus.py`, `text.py`). Smoke-tested at 18 documents per source.
- [x] Editing client for OpenAI-compatible endpoints, resumable, with rejection and resampling, as concurrent requests
      or through the Batch API (`edit.py --batch`; checked against the HUIT gateway).
- [x] Diff labels: word-level edit count, measured strength, per-word AI mask, and `versions.parquet`
      (`edit.py --consolidate`).
- [x] Zero-shot scores: log-likelihood, Fast-DetectGPT (analytic), Binoculars (`score_lm.py`).
- [x] Supervised scores: fine-tuned ModernBERT, sentence embeddings + logistic regression (`score_classifier.py`).
- [x] Length-binned conformal p-values against a per-source null (`calibrate.py`) and Efron empirical null (`stats.py`).
- [x] Global tests (HC, HC+, min-p, Stouffer) with Monte Carlo thresholds; BH and Bonferroni recovery (`stats.py`).
- [x] Gaussian simulation, empirical phase diagrams per source in window and whole-document modes, plots (`phase.py`).
- [x] Unit tests for the text and statistics helpers (`tests/`).

## To implement

Weeks 1-3: detection
- [ ] Settle every item in review.md.
- [ ] Choose editors and serve the open ones with vLLM on RunPod (editor list partly decided, review.md item 14).
- [ ] Per-editor coverage in `edit.py`: full test split at every strength, a sampled set of training edits.
- [x] Trial run with `gpt-6-luna` (reasoning off) through the HUIT gateway
      (`https://go.apis.huit.harvard.edu/ais-openai-direct-comdev/v2/`, key `OPENAI_TEST_KEY` sent as the `api-key`
      header, $10/month quota) on the 8-source smoke corpus: 13,055 edits, about $0.18 of quota.
- [x] Style comparison study (astra vs sol vs luna, $7.55 of quota): sol at reasoning low edits like astra on surface
      measures at a fifth of the cost; luna differs. See [style-comparison.md](style-comparison.md).
- [x] Same study with Claude Opus / Sonnet / Haiku 5.5 as subagents: neither smaller Claude model edits like Opus.
- [ ] Detector-transfer test between editors (needs the RunPod detectors).
- [ ] Batch tails can stall (the trial's batch sat at 7,869 / 8,167 for an hour); `--batch` waits indefinitely.
- [ ] Downstream steps with the real detectors on RunPod (the multi-source plumbing ran once with tiny stand-in models).
- [ ] Check the third-party mirrors, PERSUADE/ELLIPSE overlap, and release license terms (review.md items 3-5).
- [ ] RunPod setup: environment, storage for `data_dir`, how each step is launched.
- [ ] PC-adjusted HC (method still to be specified).
- [ ] Pangram and any other closed-source detectors.
- [ ] The week-2 "estimators" (which quantities to estimate, e.g. the edited fraction, still to be specified).
- [ ] Confidence intervals on power and recovery rates.
- [ ] Freeze the phase diagrams (week 3).

Week 4: training
- [ ] Training corpus: about 2.5B tokens of C4, one edit per sentence at the chosen settings.
- [ ] Weights: calibrated P(human | x) / local fdr per sentence; document-level weights as in Drayson et al. (2025);
      oracle mask from the true labels.
- [ ] Inverse-propensity correction for the editor's non-random choice of sentences.
- [ ] 125M Llama-style Transformer++ (RoPE, SwiGLU, RMSNorm) pretrained with per-token weights. Five arms: human only,
      unweighted mixture, document-level weights, sentence-level weights, oracle; one setting per phase-diagram region.
- [ ] Evaluation: perplexity on held-out human text, diversity of generations.
- [ ] Stretch: fine-tune Qwen3 1.7B, 4B or 8B.

Week 5
- [ ] Report, presentation, website.
- [ ] Dataset release.

Open question from the proposal
- [ ] Is the problem symmetric with the reverse case, human edits in AI text (Kashtan & Kipnis 2024)?
