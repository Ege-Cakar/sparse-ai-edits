# To do

Everything implemented so far is preliminary. The C4-only version ran end to end on a smoke test (48 C4 documents,
a local gemma3:1b editor, tiny stand-in detector models); the multi-source corpus step has been run on its own. No
setting is final. Open decisions are in
[review.md](review.md).

## Implemented (preliminary)

- [x] Corpus: eight human-text sources (FineWeb-Edu, peS2o, PERSUADE, ELLIPSE, XSum, OpenReview reviews, Enron
      emails, WritingPrompts), whole documents of 3-512 sentences, document-level splits filled to sentence targets
      per source (`sources.py`, `corpus.py`, `segment.py`, `text.py`; review.md items 1-13). Built 2026-10-30 on a
      RunPod B200: ftfy and source repairs, exact-repeat documents dropped, sentences and paragraph breaks from
      Qwen3.5-122B-A10B with SaT fallback (88% of sentences split by Qwen; replies cached in segment_cache.jsonl):
      313,249 sentences in 14,654 documents, 192,522 editable in train/val/test; sha256 8fba77d9. Frozen for editing.
- [x] Editing round 1 (2026-10-31, RunPod, ~$65): gpt-oss-120b, Gemma 4 31B, Qwen3.5-122B (held out), each with
      the "edit" and "light" wordings over every editable train/val/test sentence (192,522 records per pass, one per
      sentence, in data/edits/). Accepted after up to 3 attempts, edit / light: gpt-oss 93.8% / 93.2%,
      Gemma 94.5% / 90.1%, Qwen 92.1% / 88.1%; the rest stay human (too_long, sentence_count_changed, unchanged).
- [ ] Editing round 2: GLM 5.3 Flash ("edit" and "light"), then the API editors. GLM on 4 H200s (2026-10-31, $17,
      stopped): ~830 edits/min (a pass ~3.9 h, ~$82), prefill-bound. Its hybrid (Mamba) layers make vLLM 0.32 cache
      prefixes in 1,152-token blocks, so most documents are re-read for every sentence (prefix hit rate 20%);
      --max-num-batched-tokens 32768 crashes at startup (illegal memory access), 8192 works. Smoke test (1,000 edits):
      93.5% accepted, no reasoning tokens at reasoning_effort low.
      GLM done 2026-11-01 on kempner_requeue (slurm/edit_glm.sbatch; 4 H200s per pass, ~1,500-2,000 edits/min,
      ~2-2.5 h of running each across 6-8 preemptions per job, no records lost). Accepted, edit / light: 94.7% / 90.8%.
      Claude Haiku 5.5 done 2026-11-01 (Anthropic Message Batches, document cached; $18.32 / $17.65, 70% / 83% of
      input tokens from cache): 87.8% / 77.1% accepted (light: 15.4% left unchanged after 3 attempts).
      gpt-6-luna: 1,000-sentence pilot per pass ($0.06 / $0.07, 92.8% / 86.6% accepted; prompt caching gave ~0%
      hits in the Batch API, even above 1,024 tokens). Full passes running from 2026-11-01 (~$24 projected), one
      60 MB batch at a time: OpenAI queues at most 20M prompt tokens for luna, so larger batches fail
      (token_limit_exceeded, not billed); ~360 requests/min, so ~9 h per pass.
- [x] Editing client for OpenAI-compatible endpoints, resumable, with rejection and resampling, as concurrent requests
      or through the Batch API (`edit.py --batch`; checked against the HUIT gateway).
- [x] Diff labels: word-level edit count, measured strength, per-word AI mask, and `versions.parquet`
      (`edit.py --consolidate`).
- [x] Zero-shot scores: log-likelihood, Fast-DetectGPT (analytic), Binoculars (`score_lm.py`).
- [x] Supervised scores: fine-tuned ModernBERT, sentence embeddings + logistic regression (`score_classifier.py`).
- [x] Length-binned conformal p-values against a per-source null (`calibrate.py`) and Efron empirical null (`stats.py`).
- [x] Global tests (HC, HC+, min-p, Stouffer) with Monte Carlo thresholds; BH and Bonferroni recovery (`stats.py`).
- [x] Gaussian simulation, empirical phase diagrams per source in window and whole-document modes, plots (`phase.py`).
- [x] HC under dependence: innovated and PC-adjusted HC, Monte Carlo thresholds under a covariance, the oracle
      recovery threshold (`hc_variants.py`, Rui). Not yet used by `phase.py`.
- [x] Optional masked-LM control editor with exact k (`edit_mlm.py`, `configs/mlm_control.yaml`, Rui); runs only
      when named with `--editors`.
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
- [ ] Wire innovated / PC-adjusted HC into `phase.py`: covariance from calib windows, choice of K (review items 58-61).
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
