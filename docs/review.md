# Experimental decisions to review

Every value in `configs/default.yaml` and every choice in `src/sparse_edits/` is preliminary. They were picked only
so the pipeline runs end to end. Every item below is TBD until the team signs off on it. Each item gives the
preliminary choice, where it lives, and what to check.

## Sources

1. Which sources. TBD.
   - Preliminary: eight sources (`corpus.sources`, loaders in `sources.py`): C4 web pages; PERSUADE 2.0 essays by US
     students in grades 6-12; ELLIPSE essays by English language learners in grades 8-12; XSum BBC news articles;
     arXiv abstracts; OpenReview peer reviews; Enron emails (AESLC); PG-19 books.
   - Review: apart from C4, each source is a setting where AI detection is used in practice: classrooms, learner
     writing, journalism, academic publishing, peer review, workplace email, books. ELLIPSE matters because detectors
     have been reported to flag non-native writing as AI (Liang et al. 2023). Considered and not added: CNN/DailyMail
     (Daily Mail articles carry page boilerplate), ASAP essays (redistribution terms unclear), PeerRead (only on
     GitHub as per-paper JSON), patents (HUPD) and court opinions (not checked in depth), Reddit / WritingPrompts and
     product reviews (web text).
2. Date cutoff. TBD.
   - Preliminary: `max_year: 2019` (`corpus.defaults`), applied where documents carry dates: arXiv (first-submission
     year from the id) and OpenReview (venue year, e.g. ICLR 2018 and 2019). The rest predate it by construction:
     C4 (April 2019 crawl), XSum (2010-2017), Enron (2000-2002), PG-19 (published before 1919). PERSUADE and ELLIPSE
     carry no dates; both were released through Kaggle competitions before ChatGPT (not verified here).
   - Review: the proposal allows pre-2020, pre-2021 or pre-2022 text. arXiv abstracts are the version in a 2021
     snapshot, so a paper first submitted in 2019 can carry an abstract revised in 2020-2021. Pre-2020 text can
     contain pre-LLM tool edits (spell checkers, Grammarly). C4 is not purely human-written: the smoke run sampled
     article-spinner output ("a combined bag of outcomes, including hyperlinks to weight gain").
3. Licenses. TBD.
   - Preliminary: PERSUADE and ELLIPSE CC BY-NC-SA 4.0; the OpenReview dataset CC BY-NC 4.0; arXiv metadata CC0;
     C4 ODC-BY; XSum and AESLC "unknown" on the Hub; PG-19 books are public domain.
   - Review: a released dataset of edits inherits the non-commercial and share-alike terms of PERSUADE, ELLIPSE and
     OpenReview. Check before release.
4. Access and mirrors. TBD.
   - Preliminary: PERSUADE from the authors' Google Drive link (downloaded with `gdown`, cached in `data_dir/raw`);
     ELLIPSE from the authors' GitHub; the rest from the Hugging Face Hub: `allenai/c4`, `EdinburghNLP/xsum`,
     `gfissore/arxiv-abstracts-2021`, `nhop/OpenReview`, `Yale-LILY/aeslc`, `emozilla/pg19` (a parquet copy of
     `deepmind/pg19`, whose loading script current `datasets` cannot run).
   - Review: `gfissore`, `nhop` and `emozilla` are third-party uploads; check them against the originals.
5. Overlap between sources. TBD.
   - Review: PERSUADE and ELLIPSE come from US school writing collected by the same group; check for shared essays
     before treating one as a held-out domain for the other.

## Corpus

6. Sampling within a source. TBD.
   - Preliminary: streaming sources use a seeded shuffle buffer (`shuffle_buffer`: 10,000 documents; 200 for PG-19,
     whose rows are whole books), which shuffles shard order and then draws through the buffer, so it is an
     approximate shuffle. PERSUADE and ELLIPSE are loaded whole and shuffled exactly.
7. Split sizes. TBD.
   - Preliminary: C4 2000 train / 200 val / 2000 calib / 1000 test documents; every other source 500 / 50 / 500 / 250
     (`corpus.sources.*.n_docs`).
   - Review: test size sets the Monte Carlo error of every power estimate; calib size sets the resolution of the
     p-values (about calib sentences / 10 null sentences per length bin). Equal document counts give unequal
     sentence counts across sources.
8. Split assignment and leakage. TBD.
   - Preliminary: within each source, splits are filled in stream order, so they are disjoint by document. No
     near-duplicate removal. Reviews of the same paper are separate documents and can land in different splits.
   - Review: C4 has near-duplicate pages (templates, syndicated articles). A near-duplicate across train and test
     leaks into the classifiers; across calib and test it makes p-values dependent.
9. Sentence segmentation. TBD.
   - Preliminary: NLTK Punkt (`punkt_tab`) within paragraphs (`text.sentences`). Paragraphs are split on blank lines
     for PERSUADE, ELLIPSE, arXiv and PG-19 (single newlines there are line wrapping) and on newlines for C4, XSum,
     OpenReview and Enron (`sources.SOURCES`).
   - Review: Punkt keeps a sentence together at single-letter initials ("d. E"), and it also decides the editor's
     "exactly one sentence" check. Bullet lists in reviews become sentences. Alternatives: spaCy, pySBD, blingfire.
10. Word definition. TBD.
    - Preliminary: lowercased `\w+(?:['’]\w+)*` tokens (`text.words`). Punctuation-only and case-only changes are not
      edits; hyphenated words count as two words; numbers count as words.
    - Review: this one definition sets L, k, the edit count, and the length bins.
11. Sentence eligibility. TBD.
    - Preliminary: keep sentences with 8 to 60 words; drop the rest from the document and reindex (`min_words`,
      `max_words`). A document is its sequence of eligible sentences, and n counts only those.
    - Review: dropping short sentences changes documents (headings and list items disappear) and changes what n means.
12. Document eligibility. TBD.
    - Preliminary: 8 to 64 eligible sentences (`min_doc_sentences`, `max_doc_sentences`); 4 or more for arXiv
      abstracts and Enron emails, which are short.
    - Review: the upper cap bounds editing cost but removes long documents from whole-document mode. Short-document
      sources contribute nothing to windows of n = 16 or 32.
13. Per-source document construction. TBD.
    - Preliminary: OpenReview, one review per document: summary, main review, strengths and weaknesses, questions,
      limitations and review summary joined in that order, field-name prefixes removed, review title dropped. PG-19,
      one passage per book: 20,000 characters on each side of the midpoint, cut paragraphs at both ends dropped, then
      the first 64 eligible sentences (`truncate`). Enron documents are keyed by a hash of the body.

## Editing

14. Editor models. Partly decided 2026-10-18 by Ege.
    - Training editors: gpt-6-luna (reasoning off), gpt-6.1-sol (reasoning low), Claude Haiku / Sonnet / Opus 5.5,
      Kimi K3, GLM 5.3, DeepSeek V4.1 Flash (1017). gpt-6-astra gets a 5,000-edit test-split check paired with sol's
      edits. Qwen 3.5 (397B-A17B) is the proposed held-out editor. Not yet in the config (`edit.editors: []`).
    - Preliminary: an open 8B-class editor on RunPod for 2024-era diversity.
    - Review: whether to hold out Qwen 3.5 (an unseen lab) and also hold out other versions of training editors (an
      idea from Ege) to test version generalization. Open-weight models through OpenRouter can be served by
      different providers, some with quantized weights: pin a provider and record it. Turn reasoning off where the
      API allows (Kimi K3 untested). An editor from the same family as the zero-shot scoring models (item 28)
      confounds those detectors.
15. Prompt. TBD.
    - Preliminary: system message "You are an AI writing assistant that edits people's writing."; the user message
      asks to polish the sentence, change exactly k of its n words, keep the meaning, keep one sentence, and reply
      with only the sentence (`edit.SYSTEM`, `edit.PROMPT`).
    - Review: the prompt defines what an "AI edit" is in this study. "Polishing" pushes toward AI style; a neutral
      "change k words" instruction gives a different signal. Several prompt styles are an option. One fixed template
      can also become an artifact that classifiers learn.
16. Editing context. TBD.
    - Preliminary: the editor sees only the sentence; the masked-infilling editor matches this by default
      (`context: false`, item 65), so editors differ in how they edit, not in what they see.
    - Review: real AI editing sees the paragraph or document, which changes word choice and consistency with neighbors.
17. Strength definition and grid. TBD.
    - Preliminary: target r in {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}; k = max(1, round(r L)) (`edit.target_strengths`,
      `edit.n_target`).
    - Review: k >= 1 puts a floor of 1/L on the strength of short sentences, so a low target means different things
      at different L.
18. Edits per sentence. TBD.
    - Preliminary (budget plan, 2026-10-18): on the test split, every editor edits every sentence at all 7 strengths
      once. On train/val, each training editor edits a random sample of 50,000 (sentence, strength) pairs. A second
      sample (decided 2026-10-19 by Ege): the cheap editors redo the whole test split; the expensive ones (Sonnet,
      sol, Opus, Kimi K3, Qwen 3.5, astra) redo one shared subset of 5,000 test pairs. Not yet
      implemented: `edit.py` currently edits every sentence of `edit.splits` `edit.reps` times for every editor.
    - Review: the proposal aims for up to about 100 versions per sentence in the detection set (10 editors x 7
      strengths gives 70); whether training samples are drawn independently per editor or shared across editors
      (same sentences and strengths, a paired design). `edit --dry-run` prints request counts.
19. Which splits are edited. TBD.
    - Preliminary: train, val, test; calib stays human-only (`edit.splits`).
20. Sampling parameters and reasoning effort. TBD.
    - Preliminary: set per editor in `params`; the smoke tests used temperature 0.7, and for `gpt-6-luna`
      `reasoning_effort: none` with `max_completion_tokens: 256` (it rejects `max_tokens`). No request seeds are sent,
      so editor outputs are not reproducible.
    - Review: reasoning effort changes both cost and behavior. On one test sentence (k = 4 of 21 words), `gpt-6-luna`
      used 556 reasoning tokens at its default effort ($0.00031 per edit, changed 4 words), 195 at `low`
      ($0.000125, 3 words) and 0 at `none` ($0.000023, more than 4 words). Accepted values: none, low, medium, high,
      xhigh.
21. Output cleaning. TBD.
    - Preliminary: strip whitespace and one pair of wrapping quotes that the original did not have (`edit.clean`).
22. Rejection and resampling. TBD.
    - Preliminary: reject output that is empty, not exactly one Punkt sentence, or has unchanged words; resample up
      to 3 times (`edit.max_attempts`). Still-rejected outputs stay in `edits.parquet` and are left out of
      `versions.parquet`.
    - Review: nothing checks that the edit changed about k words. Left-out edits make documents incomplete, and
      `phase.py` drops those documents (item 51), which can select against sentences that are hard to edit.
23. Measuring the edit. TBD.
    - Preliminary: word-level Levenshtein distance (insertions + deletions + substitutions) on item-10 words;
      actual_r = distance / L_original, which can exceed 1 (`text.word_diff`).
    - Review: alternatives normalize by max(L_original, L_edited) or measure at the token or character level.
24. Diff labels. TBD.
    - Preliminary: `ai_word_mask` marks inserted and substituted words of the edited sentence; deletions leave no mark.
25. Strength used in the analysis. Partly decided 2026-10-21 by Ege: large editors track k closely enough;
    exact k is not required.
    - Preliminary: phase diagrams group edits by target strength; the measured strength is recorded per cell
      (`mean_actual_r`).
    - Review: editors do not hit k exactly. Options: target strength, binned measured strength, or resampling edits
      outside a tolerance around k.
26. Editing learner text. TBD.
    - Review: a "polish" prompt on student and learner essays fixes spelling and grammar errors, and those fixes count
      as word edits, so a given strength means something different on PERSUADE and ELLIPSE than on edited prose.

## Detectors

27. Scoring context. TBD.
    - Preliminary: every sentence is scored in isolation.
    - Review: scoring with the preceding human sentences as context gives the LM scorers more information but makes a
      sentence's score depend on its neighbors.
28. Zero-shot scoring models. TBD.
    - Preliminary: observer `tiiuae/falcon-7b`, performer `tiiuae/falcon-7b-instruct` (`lm_scores`), the Binoculars
      defaults and the Fast-DetectGPT script defaults.
    - Review: the Fast-DetectGPT README reports Llama3-8B / Llama3-8B-Instruct beating the Falcon pair; that pair
      would overlap with any Llama editor.
29. Fast-DetectGPT variant. TBD.
    - Preliminary: analytic sampling discrepancy, sampling model = observer, scoring model = performer.
30. Binoculars averaging. TBD.
    - Preliminary: performer log-PPL and observer-to-performer cross-entropy are averaged over the same predicted
      positions (the paper's definition). The reference code averages the cross-entropy over one extra position.
31. Perplexity feature. TBD.
    - Preliminary: mean token log-prob under the observer. No BOS token is prepended, so the first token is never
      scored (same as the reference implementations). Inputs are truncated at 128 tokens.
32. Classifier training data. Partly decided 2026-10-18 by Ege: every editor except the held-out ones goes into
    classifier training, frontier models included; generalization is tested on held-out editors.
    - Preliminary: train-split human sentences of all sources pooled, plus 1 random accepted edit of each
      (`classifier.edits_per_human`) from non-held-out editors, uniform over editors, strengths and reps; val is built
      the same way for early stopping.
      Editors with `train_classifier: false` (the masked-infilling control, item 62) are excluded as well.
    - Review: mixing all strengths (full rewrites included) vs emphasizing weak edits; a balanced training set vs the
      sparse test setting; pooled vs per-source classifiers, and holding out whole sources to test transfer.
33. Fine-tuned classifier. TBD.
    - Preliminary: `answerdotai/ModernBERT-base`, lr 3e-5, 2 epochs, batch 64, max 128 tokens, best epoch by val loss,
      no hyperparameter search. Score = logit(edited) - logit(human).
34. Embedding baseline. TBD.
    - Preliminary: `sentence-transformers/all-mpnet-base-v2`, normalized embeddings, `LogisticRegressionCV(Cs=10, cv=5)`.
    - Review: semantic embedders are trained to ignore paraphrase, so they may be a poor fit for edit detection; a
      style embedding may match the proposal better.
35. Closed-source detectors. TBD.
    - Preliminary: not implemented. Plan for Pangram: score whole phase-diagram documents (billing is per started
      100-word block per text, so single sentences are uneconomical), on a subset of cells, plus human-only documents
      for its null; budgeted in the local budget doc.
    - Review: Pangram returns overlapping windows with scores and character ranges, not sentence scores, so a
      sentence score has to be derived (max or mean over windows covering the sentence). Its scores on edited
      documents depend on the neighboring sentences, unlike every other detector here, which scores sentences in
      isolation (item 27). Which editor and cells to spend it on.

## Calibration

36. Null set. TBD.
    - Preliminary: human sentences of the calib split from the same source as the tested sentence (`calibrate.py`).
    - Review: a per-source null assumes the domain is known at test time; deployment usually has no domain label.
      The alternative is one pooled null, which mixes domains with very different score distributions.
37. Length conditioning. TBD.
    - Preliminary: 10 quantile bins of the tested sentence's n_words, computed on the null; test lengths are clipped
      to the null range (`calibrate.length_bins`).
    - Review: an edited sentence is compared with human sentences of its own (edited) length. Alternatives: token
      length, finer bins, regression on length.
38. p-value type. TBD.
    - Preliminary: randomized conformal p-values, exactly Uniform(0, 1) under exchangeability and never 0 or 1
      (`stats.conformal_pvalues`).
    - Review: non-randomized p-values are conservative and can equal 1, which sends the Stouffer sum to -inf.
39. Score orientation. TBD.
    - Preliminary: higher = more AI-like for every score; Binoculars is negated (`calibrate.HIGHER_IS_HUMAN`).
40. Empirical null. TBD.
    - Preliminary: Efron (2004) central matching, a weighted quadratic fit to log histogram counts (30 bins) over the
      central 50% of z, fit separately in each phase-diagram cell on all z in that cell (`stats.empirical_null`,
      `phase.empirical_null_central`). Rows are skipped when the fit is not concave.
    - Review: it assumes most z are null, which fails in dense cells (n = 16, theta = 0.1 edits 12 of 16 sentences).
      Alternatives: Efron's MLE method; a single fit on a mostly-null set.
41. Within-document dependence and PC-adjusted HC. TBD.
    - Preliminary: not handled; p-values within a document are treated as independent.
    - Review: author style shifts all sentences of a document together. The PC adjustment from lecture still has to
      be specified.

## Global tests and recovery

42. HC definition. TBD.
    - Preliminary: Donoho-Jin (2004) form with denominator sqrt(p(1 - p)), alpha0 = 1/2, reported as both HC and HC+
      (only p >= 1/n) (`phase.hc_alpha0`).
    - Review: at n = 16, HC+ ignores every p below 0.0625.
43. Other global tests. TBD.
    - Preliminary: min-p (as -log min p) and the Stouffer sum, sum of z / sqrt(n).
44. Test thresholds. TBD.
    - Preliminary: level 0.05 (`phase.alpha`); critical values from 100,000 Monte Carlo draws of iid Uniform(0, 1)
      p-values at each n (`phase.null_mc_reps`).
    - Review: this assumes independence within documents; the theta = inf rows report the actual type I error on
      human-only documents.
45. Support recovery. TBD.
    - Preliminary: BH at q = 0.1 and Bonferroni at 0.05 (`phase.bh_q`, `phase.alpha`); metrics are Hamming error / m
      and P(exact recovery).
    - Review: the Jin-Ke threshold that attains the recovery boundaries needs (theta, r) and is not implemented.

## Phase diagrams

46. Document unit. TBD.
    - Preliminary: per source, window mode takes one random window of n consecutive eligible sentences from each test
      document with at least n; whole mode uses whole documents grouped by n (`phase.modes`).
47. Window lengths. TBD.
    - Preliminary: n in {16, 32} (`phase.window_n`).
    - Review: few test documents have 32 eligible sentences (item 12); at small n the asymptotic boundaries are loose.
48. Sparsity grid. TBD.
    - Preliminary: theta in {0.1, ..., 0.9}, with exactly m = round(n^(1 - theta)) edited sentences per document
      (`phase.thetas`).
    - Review: at small n several theta give the same m; exact m vs Bernoulli(eps) per sentence.
49. Which sentences are edited. TBD.
    - Preliminary: m positions, uniformly random among all of the window's sentences (`phase.place_edits`). A
      chosen sentence whose edit was rejected stays human (decided 2026-10-21 by Ege), so that document has fewer
      than m edits; metrics use the sentences actually edited. Seeded per cell (`phase.cell_rng`: source, mode,
      n, editor, strength, theta), so every detector sees the same windows and the same edited positions.
    - Review: real editing is not uniform; the proposal's propensity correction applies here as well as in training.
50. Edit composition within a document. TBD.
    - Preliminary: one editor and one target strength per document; each sentence uses one fixed random rep per
      (editor, strength), reused across theta, chosen by a hash of (seed, editor, strength, sentence, rep) so the
      choice does not depend on which other rows exist.
51. Missing versions. TBD.
    - Preliminary: documents with a missing human p-value are dropped. A rejected edit never drops a document or a
      sentence: the human sentence stays in place, counted as `n_kept_human`. The first rule (drop a document on
      any missing edit) kept about 0.91^16 = 22% of n = 16 windows at a 9% rejection rate.
    - Review: report `n_kept_human` per cell next to power; at high rejection rates a cell's realized m falls
      below its nominal m.
52. Mapping onto the theory axes. TBD.
    - Preliminary: mu_eff = median z of edited test sentences; r_eff = max(mu_eff, 0)^2 / (2 log n). Only window mode
      is plotted, since whole mode has no single n.
    - Review: the proposal's strength (fraction of words edited) and the phase diagram's r (signal strength) are
      different quantities. Real alternatives are not Gaussian shifts, so r_eff is a summary, not a derived quantity.
      Options: mean vs median, or matching on single-test power.
53. Gaussian simulation. TBD.
    - Preliminary: r in {0.1, ..., 4.0}, 2000 simulated documents per cell, at each window n (`phase.sim_r`,
      `phase.sim_docs`), each cell with its own seed (`cell_rng`).
54. Uncertainty. TBD.
    - Preliminary: one base seed (0), with per-cell seeds derived from it; no confidence intervals on power or
      recovery rates.
    - Review: binomial standard errors, or a bootstrap over documents.
55. Reporting across sources. TBD.
    - Preliminary: every phase-diagram row carries its source; nothing is aggregated across sources.
    - Review: whether and how to combine sources into one diagram (equal weight per source vs per document).

## Training (week 4)

56. Model size. Decided 2026-10-18 by Ege.
    - 125M parameters, as in the proposal. A ~600M model (the size of the smallest deployed Qwen3 model) was
      considered the same day and set aside for cost: at Chinchilla-optimal tokens it multiplies pretraining compute
      and training-corpus edits by about 5.
57. Token budget. Decided 2026-10-18 by Ege, tentatively.
    - 2.5B tokens, the Chinchilla-optimal budget for 125M (about 20 tokens per parameter). Money goes to the dataset
      rather than more training ("the dataset feels more important so we shouldn't spend more money on training").

## HC under dependence (`hc_variants.py`)

Simulation at p = 32 (the window size), 4000 windows, level 0.05, theta = 0.5 (m = 6); notation as in Lecture 4.

| Dependence | Statistic | Type I error with phase.py's i.i.d. threshold | Power (exact null threshold) |
|---|---|---|---|
| style factor, Sigma = 11' + I | plain HC | 0.120 | 0.16 at r = 0.5 |
| | PC-adjusted HC, K = 1 | 0.048 | 0.47 at r = 0.5 |
| neighbors, tridiagonal rho = 0.4 | plain HC | | 0.42 at r = 0.3 |
| | innovated HC | | 0.64 at r = 0.3 |

58. Estimating Sigma for a window. TBD.
    - Preliminary: not wired into phase.py. `calib_window_z` is a stub: calib sentences are the conformal null,
      so their own z-scores are not available. Options: split calib in two (null / covariance), or cross-fit.
    - Review: sample covariance vs Toeplitz (stationary in sentence position) vs banded; direct inversion is
      fine when the number of calib windows >> p, else a sparse precision estimator (`precision_sparse` stub).
59. Number of principal components K, and where PC adjustment applies. TBD.
    - Finding: PC adjustment removes signal when edits are dense. Style factor sigma_a^2 = 0.25, r = 0.3, power of
      plain vs PC-adjusted HC (K = 1): p = 16: theta 0.1: 0.54 vs 0.11, 0.3: 0.35 vs 0.18, 0.5: 0.21 vs 0.18,
      0.7: 0.14 vs 0.16; p = 32: theta 0.1: 0.67 vs 0.14, 0.5: 0.25 vs 0.27, 0.7: 0.16 vs 0.20. Use it for
      theta >= 0.5 only, or plot both.
    - Preliminary: K passed explicitly (K = 1 for one per-document style factor, where PC adjustment equals
      subtracting the window mean). `choose_K` is a stub: scree / eigen-gap, Marchenko-Pastur edge, or the K
      that restores the null type I error on held-out calib windows.
60. Thresholds after adjustment. TBD.
    - Preliminary: `null_thresholds_dependent` simulates HC on the adjusted p-values under N(0, Sigma-hat)
      (parametric bootstrap). The i.i.d. thresholds in phase.py over-reject whenever windows share a factor
      (0.120 instead of 0.05 in the simulation above), which also bears on items 41 and 44.
    - Review: the 0.120 is for a simulated factor of variance 1, a hypothesis about real text. Measure it first:
      the theta = inf rows of phase.py give the actual type I error of plain HC on human-only windows.
61. Oracle recovery threshold. TBD.
    - Preliminary: `threshold_select` implements S-hat = {X_j > sqrt(2 q* log p)}, q* from Lecture 3,
      Theorem 2; it needs (theta, r), so in the empirical diagrams it is a benchmark at r = r_eff (item 45).

## Masked-infilling editor (`edit_mlm.py`)

Optional control arm, run only when explicitly requested (`configs/mlm_control.yaml`, `--editors roberta_infill`).
Decided 2026-10-21 by Ege: it sits outside the LLM-editing setting, and large editors track k closely enough
that exact k is not needed for the main study.

Pilot: 150 dev test sentences, target r in {0.1, 0.2, 0.3, 0.5}, one rep. Signal proxy: change in GPT-2 mean token
log-prob, edited minus original (positive = more predictable, the direction expected of AI polishing). GPT-2 is a
cheap stand-in for the Falcon scorers (item 28) and should be rechecked with them.

| positions | r = 0.1 | 0.2 | 0.3 | 0.5 | share more predictable at r = 0.1 |
|---|---|---|---|---|---|
| uniform | -0.19 | -0.34 | -0.44 | -0.61 | 29% |
| plausible | +0.05 | +0.03 | +0.06 | -0.01 | 59% |
| mode_seeking | +0.18 | +0.22 | +0.23 | +0.12 | 74% |
| mode_seeking + protect (default) | +0.19 | +0.22 | +0.17 | -0.04 | 81% |

Standard errors are 0.02 to 0.06. k is exact for every row except protect at r >= 0.2 (97-99%), where a sentence
has fewer editable words than k.

62. Position choice. TBD.
    - Preliminary: `positions: mode_seeking`, weight q_i / p_i (best substitute over original), `position_power: 1`.
    - Review: uniform positions force changes where the original is the only good word, which lowers likelihood and
      leaves grammar errors ("Police was alerted"): detectors would measure broken text, not AI style. mode_seeking
      edits where the human chose an unusual word and swaps in the model's preferred one, the pull toward the mode
      that the proposal's motivation describes. It also makes the positions non-uniform within a sentence, which is
      the selection effect of item 49 at word level.
63. Content protection. TBD.
    - Preliminary: `protect: true`. Names (capitalized mid-sentence), multi-piece rare words, and words inside URLs or
      e-mail addresses are never edited; a substitute may not repeat a 4+ letter word of the sentence.
    - Review: without it mode_seeking targets rare content words first ("Loglines" -> "They", "scavengers" ->
      "fish") and repeats words ("squid-spawning squid"); both change meaning, which AI polishing does not.
64. Strength range for this editor. TBD.
    - Review: the likelihood gain saturates by r = 0.3 and turns negative at r = 0.5. After the unusual words are
      fixed, further forced changes are neutral or worse, so k/L is not monotone in signal strength. Suggest
      target r <= 0.3 for this editor and placing cells by measured r_eff (item 52), not by target strength.
    - Preliminary: per-editor `target_strengths: [0.05, 0.1, 0.2, 0.3]` in the config, overriding the global grid.
      Per-sentence gain is uncorrelated with k/L and k/sqrt(L) (|corr| < 0.05) in the pilot.
65. Fill rule. TBD.
    - Preliminary: top-1 substitute, positions filled one at a time on the current text in random order, the
      sentence alone (`context: false`, matching item 16). Deterministic given the seed, unlike API editors
      (item 20). The pilot above used `context: true`; rerun it with the default before relying on its numbers.
66. MLM. TBD.
    - Preliminary: `roberta-base` (125M parameters). Cost: about 5 jobs/s on an Apple M-series GPU; a test split
      (about 300k jobs at 7 strengths x 2 reps) needs a CUDA GPU or a smaller grid.
