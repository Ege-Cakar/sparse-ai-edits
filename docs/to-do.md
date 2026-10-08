# To do

Everything implemented so far is preliminary. It runs end to end on a smoke test (48 C4 documents, a local
gemma3:1b editor, tiny stand-in detector models), but no setting is final. Open decisions are in
[review.md](review.md).

## Implemented (preliminary)

- [x] Corpus: C4 streaming, sentence segmentation, eligibility filters, document-level splits (`corpus.py`, `text.py`).
- [x] Editing client for OpenAI-compatible endpoints, resumable, with rejection and resampling (`edit.py`).
- [x] Diff labels: word-level edit count, measured strength, per-word AI mask, and `versions.parquet`
      (`edit.py --consolidate`).
- [x] Zero-shot scores: log-likelihood, Fast-DetectGPT (analytic), Binoculars (`score_lm.py`).
- [x] Supervised scores: fine-tuned ModernBERT, sentence embeddings + logistic regression (`score_classifier.py`).
- [x] Length-binned conformal p-values (`calibrate.py`) and Efron empirical null (`stats.py`).
- [x] Global tests (HC, HC+, min-p, Stouffer) with Monte Carlo thresholds; BH and Bonferroni recovery (`stats.py`).
- [x] Gaussian simulation, empirical phase diagrams in window and whole-document modes, plots (`phase.py`).
- [x] Unit tests for the text and statistics helpers (`tests/`).

## To implement

Weeks 1-3: detection
- [ ] Settle every item in review.md.
- [ ] Choose editors and serve the open ones with vLLM on RunPod.
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
