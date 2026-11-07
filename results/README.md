# Results

The numbers behind `report/report.tex`, as CSV. Everything except `detector_full_runs.csv` and the `pretrain_*` files
is written by `report/make_tables.py` from the saved pipeline outputs in `data/full_pipeline/` (not in git). All
document and sentence results are on the test split; detector selection used only the training editors' val split.

Definitions:
- `group`: `human` (unedited test sentences; `p_below_a` is then the false-positive rate), `train` (the 5 training
  editors x 2 prompts), `qwen` (the held-out editor, both prompts).
- `p_below_a`: share of sentences whose conformal p-value (against calib human sentences of the same source and length
  bin) is below `a`, i.e. TPR at FPR `a` for edited sentences.
- Detector names: `Ensemble` and its four members are ours (fine-tuned); EditLens, GPTTrace and desklib are released
  detectors applied to single sentences; Fast-DetectGPT, Log-likelihood and Binoculars are zero-shot.
- `null` in document files: `conformal_dependent` (thresholds for within-document correlation rho, the reported one),
  `conformal` (i.i.d. thresholds), `empirical` (p-values recentered by an Efron empirical null, i.i.d. thresholds).
- `editor == "none"` and `theta == inf`: unedited documents (type I error). Otherwise m = round(n^(1 - theta)) random
  sentences carry that editor's edits.

Files:
- `sentence_tpr.csv`, `sentence_tpr_by_editor.csv`, `sentence_tpr_by_source.csv`, `sentence_tpr_by_strength.csv`:
  sentence-level rates for all 15 detectors; `r_bin` is the edit rate (word edits / words of the original).
- `document_tests.csv`: per detector, mode (`whole` documents with n >= 16, or `window` of n = 16 / 32 sentences), null,
  editor and theta, pooled over sources weighted by documents: rejection rates of HC, HC+, min-p (`reject_min_p`) and
  Stouffer (`reject_sum`) at level 0.05; true and estimated edited fraction (`eps_true`, `eps_storey`, Meinshausen-Rice
  `eps_lower` and its coverage); BH (q = 0.1) and Bonferroni recovery (Hamming error / m, exact recovery); `mu_eff`
  (median z of edited sentences).
- `document_tests_by_source.csv`: the same for whole documents and correlated thresholds, per source, with 95% intervals
  (`_lo`, `_hi`; Wilson for rates) and the estimated within-document correlation `rho`.
- `document_level_whole_document_scoring.csv`: released detectors scoring the whole document, with a document-level
  conformal test against calib human documents of the same source (level 0.05).
- `detector_search.csv`: the 29-run detector search (20% of training documents), val metrics on the training editors.
- `detector_full_runs.csv`: full training runs (all training documents), best val checkpoint per run.
- `pretrain_runs.csv`: from-scratch pretraining, one row per run (arm, size, seed): test perplexity pooled and per source,
  and the paired Delta test (mean per-token log p of the edited sentence minus its original, 30,000 test pairs), pooled
  and per source. `pretrain_delta_by_editor.csv`: Delta per editor.

Notes:
- The Falcon-7B zero-shot pairs have 1,207 whole test documents instead of 1,301: documents with a missing p-value are
  dropped.
- EditLens runs as Pangram releases it (4-bit NF4 base, bf16 compute, Pangram's text cleaning). On 600 rows of Pangram's
  test set our scores match their published scores with Pearson r 0.99999, mean |diff| 0.0008, same bucket on 597.
