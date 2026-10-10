# Experimental decisions to review

Every value in `configs/default.yaml` and every choice in `src/sparse_edits/` is preliminary. They were picked only
so the pipeline runs end to end. Every item below is TBD until the team signs off on it. Each item gives the
preliminary choice, where it lives, and what to check.

## Corpus

1. Source corpus and date cutoff. TBD.
   - Preliminary: `allenai/c4`, config `en`, the April 2019 Common Crawl snapshot (`corpus.dataset`, `corpus.config`).
   - Review: the proposal allows pre-2020, pre-2021 or pre-2022 text. C4 is not purely human-written: the smoke run
     sampled article-spinner output ("a combined bag of outcomes, including hyperlinks to weight gain"). Decide
     whether to filter spun or machine-translated pages.
2. Sampling from C4. TBD.
   - Preliminary: streaming `train` split with `shuffle(seed, buffer_size=10000)`, which shuffles shard order and
     then draws through a 10k-document buffer, so it is an approximate shuffle (`corpus.split`, `corpus.shuffle_buffer`).
3. Split sizes. TBD.
   - Preliminary: 2000 train, 200 val, 2000 calib, 1000 test documents (`corpus.n_docs`).
   - Review: test size sets the Monte Carlo error of every power estimate; calib size sets the resolution of the
     p-values (about calib sentences / 10 null sentences per length bin).
4. Split assignment and leakage. TBD.
   - Preliminary: splits are filled in stream order, so they are disjoint by document. No near-duplicate removal.
   - Review: C4 has near-duplicate pages (templates, syndicated articles). A near-duplicate across train and test
     leaks into the classifiers; across calib and test it makes p-values dependent.
5. Sentence segmentation. TBD.
   - Preliminary: NLTK Punkt (`punkt_tab`) within newline-separated paragraphs (`text.sentences`).
   - Review: Punkt keeps a sentence together at single-letter initials ("d. E"), and it also decides the editor's
     "exactly one sentence" check. Alternatives: spaCy, pySBD, blingfire.
6. Word definition. TBD.
   - Preliminary: lowercased `\w+(?:['’]\w+)*` tokens (`text.words`). Punctuation-only and case-only changes are not
     edits; hyphenated words count as two words; numbers count as words.
   - Review: this one definition sets L, k, the edit count, and the length bins.
7. Sentence eligibility. TBD.
   - Preliminary: keep sentences with 8 to 60 words; drop the rest from the document and reindex
     (`corpus.min_words`, `corpus.max_words`). A document is its sequence of eligible sentences, and n counts only those.
   - Review: dropping short sentences changes documents (headings and list items disappear) and changes what n means.
8. Document eligibility. TBD.
   - Preliminary: 8 to 64 eligible sentences (`corpus.min_doc_sentences`, `corpus.max_doc_sentences`).
   - Review: the upper cap bounds editing cost but removes long documents from whole-document mode.

## Editing

9. Editor models. TBD.
   - Preliminary: none configured (`edit.editors: []`); the entry format is in the config comment.
   - Review: which open models (vLLM on RunPod) and closed models, spread from 2024-era to current frontier; which are
     held out from classifier training (`heldout`). An editor from the same family as the zero-shot scoring models
     (item 22) confounds those detectors.
10. Prompt. TBD.
    - Preliminary: system message "You are an AI writing assistant that edits people's writing."; the user message
      asks to polish the sentence, change exactly k of its n words, keep the meaning, keep one sentence, and reply
      with only the sentence (`edit.SYSTEM`, `edit.PROMPT`).
    - Review: the prompt defines what an "AI edit" is in this study. "Polishing" pushes toward AI style; a neutral
      "change k words" instruction gives a different signal. Several prompt styles are an option.
11. Editing context. TBD.
    - Preliminary: the editor sees only the sentence.
    - Review: real AI editing sees the paragraph or document, which changes word choice and consistency with neighbors.
12. Strength definition and grid. TBD.
    - Preliminary: target r in {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}; k = max(1, round(r L)) (`edit.target_strengths`,
      `edit.n_target`).
    - Review: k >= 1 puts a floor of 1/L on the strength of short sentences, so a low target means different things
      at different L.
13. Edits per sentence. TBD.
    - Preliminary: 2 reps per (sentence, editor, strength), same for every split (`edit.reps`).
    - Review: the proposal aims for up to about 100 versions per sentence in the detection set; train and test may
      need different counts. `edit --dry-run` prints request counts.
14. Which splits are edited. TBD.
    - Preliminary: train, val, test; calib stays human-only (`edit.splits`).
15. Sampling parameters. TBD.
    - Preliminary: set per editor in `params`; the smoke test used temperature 0.7 and max_tokens 256. No request
      seeds are sent, so editor outputs are not reproducible.
    - Review: reasoning models take different parameters and can spend the token budget on reasoning.
16. Output cleaning. TBD.
    - Preliminary: strip whitespace and one pair of wrapping quotes that the original did not have (`edit.clean`).
17. Rejection and resampling. TBD.
    - Preliminary: reject output that is empty, not exactly one Punkt sentence, or has unchanged words; resample up
      to 3 times (`edit.max_attempts`). Still-rejected outputs stay in `edits.parquet` and are left out of
      `versions.parquet`.
    - Review: nothing checks that the edit changed about k words. Left-out edits make documents incomplete, and
      `phase.py` drops those documents (item 45), which can select against sentences that are hard to edit.
18. Measuring the edit. TBD.
    - Preliminary: word-level Levenshtein distance (insertions + deletions + substitutions) on item-6 words;
      actual_r = distance / L_original, which can exceed 1 (`text.word_diff`).
    - Review: alternatives normalize by max(L_original, L_edited) or measure at the token or character level.
19. Diff labels. TBD.
    - Preliminary: `ai_word_mask` marks inserted and substituted words of the edited sentence; deletions leave no mark.
20. Strength used in the analysis. TBD.
    - Preliminary: phase diagrams group edits by target strength; the measured strength is recorded per cell
      (`mean_actual_r`).
    - Review: editors do not hit k exactly. Options: target strength, binned measured strength, or resampling edits
      outside a tolerance around k.

## Detectors

21. Scoring context. TBD.
    - Preliminary: every sentence is scored in isolation.
    - Review: scoring with the preceding human sentences as context gives the LM scorers more information but makes a
      sentence's score depend on its neighbors.
22. Zero-shot scoring models. TBD.
    - Preliminary: observer `tiiuae/falcon-7b`, performer `tiiuae/falcon-7b-instruct` (`lm_scores`), the Binoculars
      defaults and the Fast-DetectGPT script defaults.
    - Review: the Fast-DetectGPT README reports Llama3-8B / Llama3-8B-Instruct beating the Falcon pair; that pair
      would overlap with any Llama editor.
23. Fast-DetectGPT variant. TBD.
    - Preliminary: analytic sampling discrepancy, sampling model = observer, scoring model = performer.
24. Binoculars averaging. TBD.
    - Preliminary: performer log-PPL and observer-to-performer cross-entropy are averaged over the same predicted
      positions (the paper's definition). The reference code averages the cross-entropy over one extra position.
25. Perplexity feature. TBD.
    - Preliminary: mean token log-prob under the observer. No BOS token is prepended, so the first token is never
      scored (same as the reference implementations). Inputs are truncated at 128 tokens.
26. Classifier training data. TBD.
    - Preliminary: train-split human sentences plus 1 random accepted edit of each (`classifier.edits_per_human`) from
      non-held-out editors, uniform over editors, strengths and reps; val is built the same way for early stopping.
    - Review: mixing all strengths (full rewrites included) vs emphasizing weak edits; a balanced training set vs the
      sparse test setting.
27. Fine-tuned classifier. TBD.
    - Preliminary: `answerdotai/ModernBERT-base`, lr 3e-5, 2 epochs, batch 64, max 128 tokens, best epoch by val loss,
      no hyperparameter search. Score = logit(edited) - logit(human).
28. Embedding baseline. TBD.
    - Preliminary: `sentence-transformers/all-mpnet-base-v2`, normalized embeddings, `LogisticRegressionCV(Cs=10, cv=5)`.
    - Review: semantic embedders are trained to ignore paraphrase, so they may be a poor fit for edit detection; a
      style embedding may match the proposal better.
29. Closed-source detectors. TBD.
    - Preliminary: not implemented (Pangram).

## Calibration

30. Null set. TBD.
    - Preliminary: human sentences of the calib split.
31. Length conditioning. TBD.
    - Preliminary: 10 quantile bins of the tested sentence's n_words, computed on the null; test lengths are clipped to
      the null range (`calibrate.length_bins`).
    - Review: an edited sentence is compared with human sentences of its own (edited) length. Alternatives: token
      length, finer bins, regression on length.
32. p-value type. TBD.
    - Preliminary: randomized conformal p-values, exactly Uniform(0, 1) under exchangeability and never 0 or 1
      (`stats.conformal_pvalues`).
    - Review: non-randomized p-values are conservative and can equal 1, which sends the Stouffer sum to -inf.
33. Score orientation. TBD.
    - Preliminary: higher = more AI-like for every score; Binoculars is negated (`calibrate.HIGHER_IS_HUMAN`).
34. Empirical null. TBD.
    - Preliminary: Efron (2004) central matching, a weighted quadratic fit to log histogram counts (30 bins) over the
      central 50% of z, fit separately in each phase-diagram cell on all z in that cell (`stats.empirical_null`,
      `phase.empirical_null_central`). Rows are skipped when the fit is not concave.
    - Review: it assumes most z are null, which fails in dense cells (n = 16, theta = 0.1 edits 12 of 16 sentences).
      Alternatives: Efron's MLE method; a single fit on a mostly-null set.
35. Within-document dependence and PC-adjusted HC. TBD.
    - Preliminary: not handled; p-values within a document are treated as independent.
    - Review: author style shifts all sentences of a document together. The PC adjustment from lecture still has to
      be specified.

## Global tests and recovery

36. HC definition. TBD.
    - Preliminary: Donoho-Jin (2004) form with denominator sqrt(p(1 - p)), alpha0 = 1/2, reported as both HC and HC+
      (only p >= 1/n) (`phase.hc_alpha0`).
    - Review: at n = 16, HC+ ignores every p below 0.0625.
37. Other global tests. TBD.
    - Preliminary: min-p (as -log min p) and the Stouffer sum, sum of z / sqrt(n).
38. Test thresholds. TBD.
    - Preliminary: level 0.05 (`phase.alpha`); critical values from 100,000 Monte Carlo draws of iid Uniform(0, 1)
      p-values at each n (`phase.null_mc_reps`).
    - Review: this assumes independence within documents; the theta = inf rows report the actual type I error on
      human-only documents.
39. Support recovery. TBD.
    - Preliminary: BH at q = 0.1 and Bonferroni at 0.05 (`phase.bh_q`, `phase.alpha`); metrics are Hamming error / m
      and P(exact recovery).
    - Review: the Jin-Ke threshold that attains the recovery boundaries needs (theta, r) and is not implemented.

## Phase diagrams

40. Document unit. TBD.
    - Preliminary: window mode takes one random window of n consecutive eligible sentences from each test document
      with at least n; whole mode uses whole documents grouped by n (`phase.modes`).
41. Window lengths. TBD.
    - Preliminary: n in {16, 32} (`phase.window_n`).
    - Review: few test documents have 32 eligible sentences (item 8); at small n the asymptotic boundaries are loose.
42. Sparsity grid. TBD.
    - Preliminary: theta in {0.1, ..., 0.9}, with exactly m = round(n^(1 - theta)) edited sentences per document
      (`phase.thetas`).
    - Review: at small n several theta give the same m; exact m vs Bernoulli(eps) per sentence.
43. Which sentences are edited. TBD.
    - Preliminary: uniformly random positions.
    - Review: real editing is not uniform; the proposal's propensity correction applies here as well as in training.
44. Edit composition within a document. TBD.
    - Preliminary: one editor and one target strength per document; each sentence uses one fixed random rep per
      (editor, strength), reused across theta.
45. Missing versions. TBD.
    - Preliminary: documents with any missing human or edited p-value are dropped from that (editor, strength) cell.
46. Mapping onto the theory axes. TBD.
    - Preliminary: mu_eff = median z of edited test sentences; r_eff = max(mu_eff, 0)^2 / (2 log n). Only window mode
      is plotted, since whole mode has no single n.
    - Review: the proposal's strength (fraction of words edited) and the phase diagram's r (signal strength) are
      different quantities. Real alternatives are not Gaussian shifts, so r_eff is a summary, not a derived quantity.
      Options: mean vs median, or matching on single-test power.
47. Gaussian simulation. TBD.
    - Preliminary: r in {0.1, ..., 4.0}, 2000 simulated documents per cell, at each window n (`phase.sim_r`,
      `phase.sim_docs`).
48. Uncertainty. TBD.
    - Preliminary: one seed (0); no confidence intervals on power or recovery rates.
    - Review: binomial standard errors, or a bootstrap over documents.

## Masked-infilling editor (`edit_mlm.py`)

Pilot: 150 dev test sentences, target r in {0.1, 0.2, 0.3, 0.5}, one rep. Signal proxy: change in GPT-2 mean token
log-prob, edited minus original (positive = more predictable, the direction expected of AI polishing). GPT-2 is a
cheap stand-in for the Falcon scorers (item 22) and should be rechecked with them.

| positions | r = 0.1 | 0.2 | 0.3 | 0.5 | share more predictable at r = 0.1 |
|---|---|---|---|---|---|
| uniform | -0.19 | -0.34 | -0.44 | -0.61 | 29% |
| plausible | +0.05 | +0.03 | +0.06 | -0.01 | 59% |
| mode_seeking | +0.18 | +0.22 | +0.23 | +0.12 | 74% |
| mode_seeking + protect (default) | +0.19 | +0.22 | +0.17 | -0.04 | 81% |

Standard errors are 0.02 to 0.06. k is exact for every row except protect at r >= 0.2 (97-99%), where a sentence
has fewer editable words than k.

49. Position choice. TBD.
    - Preliminary: `positions: mode_seeking`, weight q_i / p_i (best substitute over original), `position_power: 1`.
    - Review: uniform positions force changes where the original is the only good word, which lowers likelihood and
      leaves grammar errors ("Police was alerted"): detectors would measure broken text, not AI style. mode_seeking
      edits where the human chose an unusual word and swaps in the model's preferred one, the pull toward the mode
      that the proposal's motivation describes. It also makes the positions non-uniform within a sentence, which is
      the selection effect of item 43 at word level.
50. Content protection. TBD.
    - Preliminary: `protect: true`. Names (capitalized mid-sentence), multi-piece rare words, and words inside URLs or
      e-mail addresses are never edited; a substitute may not repeat a 4+ letter word of the sentence.
    - Review: without it mode_seeking targets rare content words first ("Loglines" -> "They", "scavengers" ->
      "fish") and repeats words ("squid-spawning squid"); both change meaning, which AI polishing does not.
51. Strength range for this editor. TBD.
    - Review: the likelihood gain saturates by r = 0.3 and turns negative at r = 0.5. After the unusual words are
      fixed, further forced changes are neutral or worse, so k/L is not monotone in signal strength. Suggest
      target r <= 0.3 for this editor and placing cells by measured r_eff (item 46), not by target strength.
      Per-sentence gain is uncorrelated with k/L and k/sqrt(L) (|corr| < 0.05) in the pilot.
52. Fill rule. TBD.
    - Preliminary: top-1 substitute, positions filled one at a time on the current text in random order, previous
      and next eligible sentence as context (`context: true`). Deterministic given the seed, unlike API editors
      (item 15).
53. MLM. TBD.
    - Preliminary: `roberta-base`. Cost: about 5 jobs/s on an Apple M-series GPU; the default config's test split
      (about 300k jobs at 7 strengths x 2 reps) needs a CUDA GPU or a smaller grid.
