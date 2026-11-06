# Experimental decisions to review

Every value in `configs/default.yaml` and every choice in `src/sparse_edits/` is preliminary. They were picked only
so the pipeline runs end to end. Every item below is TBD until the team signs off on it. Each item gives the
preliminary choice, where it lives, and what to check.

## Sources

1. Which sources. Decided 2026-10-29 by Ege.
   - Eight sources (`corpus.sources`, loaders in `sources.py`): FineWeb-Edu web pages; peS2o full-text papers;
     PERSUADE 2.0 essays by US students in grades 6-12; ELLIPSE essays by English language learners in grades 8-12;
     XSum BBC news articles; OpenReview peer reviews; Enron emails (AESLC); WritingPrompts short stories.
   - Replaced: C4 by FineWeb-Edu (higher-quality web text), arXiv abstracts by peS2o (abstracts have at most 15
     sentences, so they never form 16- or 32-sentence windows), PG-19 by WritingPrompts (pre-1919 English is far
     from current writing; no open full-text book corpus newer than PG-19 was found).
   - Review: FineWeb-Edu is selected by a classifier trained on Llama-3-70B ratings of educational value, so an
     LLM's judgment picks the human text (DCLM-baseline was avoided for the stronger version of this: its classifier's
     positives are GPT-4-generated OpenHermes data). The training corpus (week 4) is still C4, which the detection
     corpus no longer contains. ELLIPSE matters because detectors have been reported to flag non-native writing as AI
     (Liang et al. 2023). Considered and not added: CNN/DailyMail (Daily Mail articles carry page boilerplate), ASAP
     essays (redistribution terms unclear), PeerRead (only on GitHub as per-paper JSON), patents (HUPD) and court
     opinions (not checked in depth), BookCorpusOpen (no license), C4 blog pages selected by URL (spam remains).
2. Date cutoff. Decided 2026-10-29 by Ege: 2020.
   - `max_year: 2020` (`corpus.defaults`), applied where documents carry dates: FineWeb-Edu (crawl year of the Common
     Crawl dump, which bounds the page's creation), peS2o (`created`, a best-guess first-publication date) and
     OpenReview (venue year). The rest predate it by construction: XSum (2010-2017), Enron (2000-2002),
     WritingPrompts (collected for Fan et al. 2018). PERSUADE and ELLIPSE carry no dates; both were released through
     Kaggle competitions before ChatGPT (not verified here).
   - Review: GPT-3 was released in June 2020, so 2020 text can include GPT-3-assisted writing (rare then). Text of
     any year can contain pre-LLM tool edits (spell checkers, Grammarly). Web text is not purely human-written: the C4
     smoke run sampled article-spinner output ("a combined bag of outcomes, including hyperlinks to weight gain").
3. Licenses. TBD.
   - Preliminary: PERSUADE and ELLIPSE CC BY-NC-SA 4.0; the OpenReview dataset CC BY-NC 4.0; FineWeb-Edu and peS2o
     ODC-By; WritingPrompts MIT on the Hub (Reddit posts); XSum and AESLC "unknown" on the Hub.
   - Review: a released dataset of edits inherits the non-commercial and share-alike terms of PERSUADE, ELLIPSE and
     OpenReview. Check before release.
4. Access and mirrors. TBD.
   - Preliminary: PERSUADE from the authors' Google Drive link (downloaded with `gdown`, cached in `data_dir/raw`;
     SHA-256 f61319ed...); ELLIPSE from the authors' GitHub at commit b5f44696; the rest from the Hugging Face Hub at
     the revisions pinned in `sources.REVISIONS` (2026-10-29): `HuggingFaceFW/fineweb-edu`, `allenai/peS2o` (read
     as JSON, files 10-19 of `data/v2` hold the full texts; its loading script no longer runs), `EdinburghNLP/xsum`,
     `nhop/OpenReview`, `Yale-LILY/aeslc`, `euclaise/writingprompts`.
   - Review: `nhop` and `euclaise` are third-party uploads; check them against the originals.
5. Overlap between sources. TBD.
   - Review: PERSUADE and ELLIPSE come from US school writing collected by the same group; check for shared essays
     before treating one as a held-out domain for the other.

## Corpus

6. Sampling within a source. TBD.
   - Preliminary: streaming sources use a seeded shuffle buffer (`shuffle_buffer`: 10,000 documents), which shuffles
     shard order and then draws through the buffer, so it is an approximate shuffle. FineWeb-Edu's first 1,500 rows
     in build order came from several crawls (2014-2018, about 150-180 rows from each of the top five). PERSUADE and
     ELLIPSE are loaded whole and shuffled exactly.
7. Split sizes. Decided 2026-10-29 by Ege.
   - Every source gets 15,000 train / 1,500 val / 15,000 calib / 7,500 test sentences (`corpus.defaults.n_sentences`):
     each split is filled with whole documents until it reaches its target, so it overshoots by less than one
     document. Equal sentence counts give equal edit cost, test size and calib resolution per source (about 1,500 null
     sentences per length bin); the number of documents varies with document length (few for peS2o).
   - Review: power estimates in window mode use one window per test document with at least n sentences, so their
     Monte Carlo error depends on how many such documents a source has, not on its sentence count.
8. Split assignment and leakage. TBD.
   - Preliminary: within each source, splits are filled in stream order, so they are disjoint by document.
     Every source skips a document whose paragraphs and sentences exactly repeat an earlier one (`corpus.unique`;
     2026-10-30): AESLC files one Enron email in up to 8 folders, and before this 93 such emails filled 228 Enron
     documents, 69 of them in more than one split; FineWeb-Edu repeats a page across crawls. No near-duplicate
     removal. Reviews of the same paper are separate documents and can land in different splits.
   - Review: web text has near-duplicate pages (templates, syndicated articles). A near-duplicate across train and
     test leaks into the classifiers; across calib and test it makes p-values dependent.
9. Sentence segmentation. Decided 2026-10-30 by Ege.
   - Paragraphs (`text.paragraphs`) are split on blank lines for PERSUADE, ELLIPSE and peS2o (single newlines there
     are line wrapping) and on newlines for FineWeb-Edu, XSum, OpenReview, Enron and WritingPrompts (`sources.SOURCES`).
     Paragraphs of only whitespace or zero-width characters are dropped (OpenReview has some of only U+200B).
   - Before segmentation, ftfy 6.3.1 repairs encoding errors, HTML entities, ligatures, control characters and line
     breaks (`corpus.FTFY`; decided 2026-10-30 by Ege), keeping quote style and character width as written. It changed
     351 documents: "Â£" mojibake in 310 XSum articles, lone `\r` line breaks that had merged the paragraphs of 17
     ELLIPSE essays, PDF ligatures. 17 sentences keep errors ftfy cannot undo (PDF font errors such as "5 Â 5 mm" in
     peS2o, characters already lost as U+FFFD).
   - Source repairs with a known cause (decided 2026-10-30 by Ege): Enron bodies with quoted-printable leftovers
     (=20, =09, =3D, soft line breaks; about 2% of AESLC emails) are decoded (`sources.decode_qp`); PERSUADE's
     processing put a space after every period followed by a letter, visible in web addresses (35 spaced domains, 0
     unspaced), which are rejoined (`sources.join_urls`).
   - Sentences and paragraph breaks from an LLM (decided 2026-10-30 by Ege; `segment.py`, `corpus.segmenter`):
     Qwen3.5-122B-A10B-FP8 (pinned revision, vLLM 0.32.0, thinking off, greedy) gets each document in chunks of whole
     paragraphs and returns one sentence per line. A reply is accepted only if, with whitespace removed, it equals the
     chunk, except for hyphens dropped before whitespace ("in- dicates"); otherwise the chunk goes to SaT. In
     WritingPrompts, peS2o, FineWeb-Edu and XSum, whose spacing errors come from tokenization or extraction, its
     spacing and hyphen fixes are kept; elsewhere only its breaks, so essays, emails and reviews keep the writers' own
     spacing. A paragraph break is kept only where the text had a line break, and a line break inside a sentence is
     wrapping: SaT within newline-split paragraphs had cut 867 FineWeb-Edu and 869 OpenReview sentences at hard line
     wraps. Replies are cached (`segment_cache.jsonl`), so rebuilds reproduce the corpus. The `segmenter` column
     records "llm", "llm+sat", "sat" or "punkt" per sentence.
     Pilots (40 documents per source): Qwen straightened curly quotes (most rejections), so quote variants count as
     equal and sentences are rebuilt from the original characters; it sometimes left two sentences on one line (11% of
     XSum lines), so SaT's breaks inside a Qwen sentence are added where the text before them ends with . ! ? or
     ... (`segment.resplit`, "llm+sat"; decided 2026-10-30 by Ege); in peS2o it may also drop a hyphen between two
     lowercase letters ("pop-ular", `segmenter.rejoin_hyphens`; decided 2026-10-30 by Ege). Learner run-ons without
     punctuation stay one sentence (decided 2026-10-30 by Ege). Exact-repeat documents are now dropped before
     splitting (`corpus.unique`), since concurrent greedy replies to identical requests can differ. Evidence that LLMs split better than SaT is thin (SaT paper:
     SaT beat the 2024 LLM baselines it tested), so the choice is checked on a hand-labelled sample.
   - SaT fallback: wtpsplit 2.2.2 SaT `sat-12l-sm` within each paragraph (`text.indexed_sentences`), at pinned model and
     tokenizer revisions (`text.SAT_MODEL`, `text.SAT_TOKENIZER`), fp32, wtpsplit's default threshold for `-sm` models
     (0.25), sentences stripped of surrounding whitespace. The corpus is built on a GPU (on 4 CPUs SaT-12 ran about
     7 sentences/s; on an RTX 4000 Ada, 350-790). A paragraph SaT fails on is split by Punkt instead
     (`segmenter` "punkt") (wtpsplit hit an internal assertion once, on an OpenReview paragraph, in a first run
     of the comparison). On the comparison's 2,024 paragraphs the GPU build gives exactly the CPU comparison's
     sentences, and the sentences rejoin to each paragraph with no characters lost.
   - Why (Punkt, pySBD, spaCy `en_core_web_sm` senter, SaT 3- and 12-layer on 20 documents per source, 2026-10-29;
     results and scripts in `data/splitter_comparison/`): only SaT separates the section headers that peS2o's PDF
     extraction glues to the next sentence ("Demographic data District-level ..."); fragments such as "I." or
     "2003)." in peS2o: Punkt 265, spaCy 145, sat-3l 30, pySBD 26, sat-12l 23; sentences over 60 words in ELLIPSE:
     Punkt 2.2%, sat-12l 0.9%. sat-3l over-splits learner run-ons; spaCy breaks bracketed citations; pySBD and
     sat-12l keep quoted dialogue together. Seen in the examples: sat-12l split "7. || Victor ..." in a numbered list.
10. Word definition. TBD.
    - Preliminary: lowercased `\w+(?:['’]\w+)*` tokens (`text.words`). Punctuation-only and case-only changes are not
      edits; hyphenated words count as two words; numbers count as words.
    - Review: this one definition sets L, k, the edit count, and the length bins.
11. Sentence eligibility. Decided 2026-10-29 by Ege.
    - Every sentence is kept, so documents stay contiguous. Short sentences are edited and tested like any other:
      k = max(1, round(r L)) asks for one edit in a 3-word sentence even at r = 0.05 (33% of its words), as intended.
      Sentences with no words ("!", ".", "***"; 0-1.4% per source) are never sent to an editor (`edit.todo_for`,
      decided 2026-10-29) and stay human in every version.
    - Review: fragments become sentences (headings, greetings, sign-offs, scene breaks such as "***").
12. Document eligibility. Decided 2026-10-29 by Ege.
    - 3 to 512 sentences (`min_doc_sentences`, `max_doc_sentences`); documents outside the range are skipped whole,
      never cut.
    - Review: whole-document mode groups test documents by exact length; 3-sentence documents give cells where
      m = round(n^(1 - theta)) is 1 at every theta.
13. Per-source document construction. TBD.
    - Preliminary: OpenReview, one review per document: summary, main review, strengths and weaknesses, questions,
      limitations and review summary joined in that order, field-name prefixes removed, review title dropped.
      WritingPrompts, the story only (prompt dropped), with the dataset's tokenization undone (`sources.detokenize`,
      decided 2026-10-29): "``" and "''" become '"', "do n't" becomes "don't", spaces inside curly quotes removed
      (found in 65%, 65%, 71% and 13% of 5,000 stories). peS2o, the Grobid text as given: title as the first
      paragraph, then the abstract and sections with their headers. XSum, only articles from the BBC News and
      Newsbeat URL sections (decided 2026-10-29; `sources.XSUM_SECTIONS`): Sport pages are dropped (27% of the built
      XSum documents; their match reports end in automated live commentary such as "Substitution, Everton."), as are
      Newsround video pages (timestamp boilerplate), Newyddion (Welsh) and School Report. Enron and WritingPrompts
      documents are keyed by a hash of the text.

## Editing

14. Editor models. Partly decided 2026-10-18 by Ege.
    - Training editors: gpt-6-luna (reasoning off), gpt-6.1-sol (reasoning low), Claude Haiku / Sonnet / Opus 5.5,
      Kimi K3, GLM 5.3, DeepSeek V4.1 Flash (1017). gpt-6-astra gets a 5,000-edit test-split check paired with sol's
      edits. Qwen 3.5 (397B-A17B) is the proposed held-out editor. Not yet in the config (`edit.editors: []`).
    - Preliminary: an open 8B-class editor on RunPod for 2024-era diversity.
    - 2026-10-31 (Ege), first editing round, self-hosted on RunPod while API keys are pending: gpt-oss-120b and
      Gemma 4 31B (training editors), Qwen3.5-122B-A10B (held out, standing in for Qwen 3.5 397B). Each edits every
      editable train/val/test sentence once (192,522 requests). GLM 5.3 (753B) needs 8 H200s (~$42/h) and Mistral
      Large 4 ("le Chonk", 1T) 8 B200s, out of stock on 2026-10-31; both licenses (modified MIT) allow research use.
      With whole-document prompts a pass is ~381M input tokens without prompt caching (~5M unique document tokens),
      so API editors cost far more than the budget's ~100 input tokens per edit unless the provider caches prompts.
      GLM 5.3 Flash (320B, 18B active, MIT) was set up on 4 H200s and dropped the same day to stay within the RunPod
      credit ($149); to run later. Rates on 2026-10-31: gpt-oss 3,600 edits/min (H200), Gemma 4 1,950 (H200),
      Qwen3.5 1,390 (B200); after 3 attempts 7.2%, 5.4% and 8.8% of sentences stay unedited.
    - 2026-11-01 (Ege): gpt-6-luna (reasoning_effort none) and Claude Haiku 5.5 (thinking disabled) join as training
      editors, both wordings, every split, at most $50 each, through each vendor's Batch API (half price) with
      prompt caching: Anthropic requests mark the document part (after the system prompt) as a cached prefix;
      OpenAI requests carry a per-document prompt_cache_key. Neither model accepts sampling parameters, so both
      sample at their defaults. A 1,000-request pilot per editor and wording (seeded whole documents) measures
      tokens, cache hits and rejections; its records are part of the dataset, so the full run skips them.
    - 2026-11-01 (Ege): luna requests after the first 1,000 light / 25,035 edit records go through flex processing
      (`service_tier: flex`, billed at Batch rates, answered per request), which is faster than the Batch API,
      whose rounds waited on stalled last requests. Same model, prompt and settings; only the serving tier differs,
      and each response's tier is in the `.usage` log.
    - Review: whether to hold out Qwen 3.5 (an unseen lab) and also hold out other versions of training editors (an
      idea from Ege) to test version generalization. Open-weight models through OpenRouter can be served by
      different providers, some with quantized weights: pin a provider and record it. Turn reasoning off where the
      API allows (Kimi K3 untested). An editor from the same family as the zero-shot scoring models (item 28)
      confounds those detectors.
15. Prompt. Decided 2026-10-29 by Ege.
    - System message (`edit.SYSTEM`): a generic assistant prompt, "You are {product}, a helpful AI assistant..." with
      five guidelines and "Today's date is November 1, 2026."; `product` is set per editor (ChatGPT for OpenAI models
      and gpt-oss, Claude for Anthropic's). The vague-request guideline asks for a best-effort answer instead of a
      clarifying question, to avoid outputs that are questions. It is not any vendor's production system prompt.
    - User message (`edit.PROMPT`): "Text: {document}", a blank line, then 'Edit the sentence "{sentence}", please in
      the text above, to improve the writing. Change exactly {k} of its {n} words by replacing, inserting, or deleting
      words. Keep the meaning and keep it one sentence. Reply with only the edited sentence, followed by a line
      break.' PERSUADE and ELLIPSE sentences that contain anonymization placeholders (Generic_Name, STUDENT_NAME, ...)
      get "Keep the placeholder(s) ... exactly as written." appended; other sentences do not.
    - Decided 2026-10-31 by Ege, in use: two wordings (`edit.ASK`, chosen per editor with `prompt`). "edit": 'Edit the
      sentence "{sentence}", please in the text above, to improve the writing. Keep the meaning and about the same
      length, and keep it one sentence.' "light" (a second pass, every split): 'Lightly edit the sentence
      "{sentence}", please in the text above, to improve the writing, changing only what needs it. Keep the meaning
      and about the same length, and keep it one sentence.' Pilot (400 test sentences, one attempt): median share of
      words changed old / edit / light = Qwen3.5 0.40 / 0.38 / 0.24, Gemma 4 0.46 / 0.46 / 0.22, gpt-oss 0.50 / 0.50 /
      0.36; "about the same length" cut over-twice-as-long edits (Gemma 5.8% -> 2.8%) but not the strength.
    - Decided 2026-10-30 by Ege: drop "Change exactly {k} of its {n} words by replacing,
      inserting, or deleting words." A requested count is not something users ask for and may make edits unnatural
      (forced word swaps to reach k); the edit strength is instead measured afterwards (`actual_r` = word edits / n
      words, already computed by `edit.consolidate`). Open: how strengths are covered without target strengths
      (item 17).
    - gpt-oss through vLLM: vLLM puts the harmony system message first ("You are ChatGPT, a large language model
      trained by OpenAI.", knowledge cutoff 2024-06, current date, "Reasoning: low") and our system message becomes
      the developer message. Its date is pinned to 2026-11-01 (`VLLM_SYSTEM_START_DATE`) to match ours.
    - Review: one fixed template for every sentence can become an artifact that classifiers learn. "To improve the
      writing" pushes toward AI style. The target sentence is identified by quoting it; if the same sentence occurs
      twice in a document, the editor may edit either occurrence (same text, so only the context differs).
16. Editing context. Decided 2026-10-29 by Ege: the whole document.
    - The editor sees the sentence's whole original document, rebuilt from the stored sentences: a paragraph's
      sentences joined by spaces (original line wrapping is lost), paragraphs joined by the source's separator
      (`edit.documents`, `sentences.para_idx`). The document comes first in the prompt, so all requests for one
      document share a prefix that vLLM's prefix cache (and API prompt caches above their minimum length) reuse; jobs
      are sent in document order, and `--limit` takes whole documents in a seeded order.
    - Each sentence is edited against the original document, so two edited sentences in the same document never see
      each other's changes; a multi-sentence edited document is assembled afterwards (`phase.py`, items 49-50) from
      independent single-sentence edits. A real AI pass over a document would edit several sentences jointly and
      consistently.
    - Cost: per pass of one editor, about 2.5B input tokens if every request is processed from scratch, about 136M
      if each document is processed once (o200k tokens on the 2026-10-29 corpus; peS2o papers average 5,755
      tokens). API editors cache only prompts above a minimum length, and most documents are shorter.
    - Review: the masked-infilling control editor still sees only the sentence by default (`context: false`, item
      65), so it now differs from the LLM editors in what it sees.
17. Strength definition and grid. TBD.
    - Preliminary: target r in {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}; k = max(1, round(r L)) (`edit.target_strengths`,
      `edit.n_target`).
    - Review: k >= 1 puts a floor of 1/L on the strength of short sentences, so a low target means different things
      at different L.
    - 2026-10-30: with the count dropped from the prompt (item 15), each editor edits at its own natural strength, so
      the grid goes away and phase cells would be per editor, placed by measured strength (`actual_r`, `mu_eff`).
      Open: whether that covers the sparse, low-strength region; binning by measured strength selects on sentences
      (the ones an editor changes most differ from the rest); varying strength by wording instead of numbers.
18. Edits per sentence. TBD.
    - Preliminary (budget plan, 2026-10-18): on the test split, every editor edits every sentence at all 7 strengths
      once. On train/val, each training editor edits a random sample of 50,000 (sentence, strength) pairs. A second
      sample (decided 2026-10-19 by Ege): the cheap editors redo the whole test split; the expensive ones (Sonnet,
      sol, Opus, Kimi K3, Qwen 3.5, astra) redo one shared subset of 5,000 test pairs. Not yet
      implemented: `edit.py` currently edits every sentence of `edit.splits` `edit.reps` times for every editor.
    - Review: the proposal aims for up to about 100 versions per sentence in the detection set (10 editors x 7
      strengths gives 70); whether training samples are drawn independently per editor or shared across editors
      (same sentences and strengths, a paired design). `edit --dry-run` prints request counts.
19. Which splits are edited. Decided 2026-10-31 by Ege: train, val, test; calib stays human-only (`edit.splits`),
    for every editor and wording, the held-out one included.
20. Sampling parameters and reasoning effort. Decided 2026-10-31 by Ege: each model's recommended settings.
    - gpt-oss-120b: temperature 1.0, top_p 1.0 (OpenAI), `reasoning_effort: low` (the lowest it has; vLLM rejects
      none), max_tokens 4096 counting reasoning; the reasoning text is not saved. Qwen3.5-122B: thinking off, Qwen's
      non-thinking settings (temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence penalty 1.5). Gemma 4 31B:
      thinking off, Google's settings (temperature 1.0, top_p 0.95, top_k 64). Self-hosted with vLLM 0.32.0 and
      `--generation-config vllm`, so only these settings apply (`edit.editors[].params`).
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
    - Preliminary: reject output that is empty, splits or merges Punkt sentences, or has unchanged words, and (decided
      2026-10-29) essay output that changed a placeholder the prompt asked to keep (`placeholder_changed`), and
      (decided 2026-10-31) output with more than twice the original's words (`too_long`: short sentences such as
      "Thanks." or "Disadvantages:" turned into invented content, 6-8% of first attempts); resample up
      to 3 times (`edit.max_attempts`). Still-rejected outputs stay in `edits.parquet` and are left out of
      `versions.parquet`.
    - Decided 2026-10-30 by Ege: with SaT sentences (item 9), the sentence check is "the output has as many Punkt
      sentences as the original" (`sentence_count_changed`, which replaces `not_one_sentence`), so a SaT sentence that
      Punkt reads as two (quoted dialogue) can still be edited, without running SaT on every output.
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
    - 2026-10-30 (Ege asked; isolated scoring kept for the main analysis): in isolation a human sentence scores the
      same whether or not its neighbors were edited, so its conformal p-value against all-human calib sentences is
      uniform in any document, and each version is scored once and looked up when windows are assembled. With
      context, a human sentence next to edits scores differently than in calib (no valid null without knowing the
      edits), and every edit placement would need rescoring. The likely price is power on short sentences.
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
    - Decided 2026-10-30 by Ege, not yet implemented: binary (human vs AI-edited) on every accepted train edit plus
      the human sentences, weighted to balance the classes instead of subsampling one edit. Open: global class
      weights vs per-sentence weights (each human sentence 1, each of its K edits 1/K).
33. Fine-tuned classifier. TBD.
    - Preliminary: `answerdotai/ModernBERT-base`, lr 3e-5, 2 epochs, batch 64, max 128 tokens, best epoch by val loss,
      no hyperparameter search. Score = logit(edited) - logit(human).
    - Decided 2026-10-31 by Ege: search both detectors (isolated sentence; document with the target sentence marked)
      at once, over embedding models with a trained head, fine-tuned modern encoders and fine-tuned decoder LLMs, as a
      small grid on a train subsample. Every run logs val loss, AUROC and TPR at 0.1%, 1% and 5% FPR; the winner is
      the highest TPR at 1% FPR (the probability that an edited sentence gets p <= 0.01, which drives HC and
      Bonferroni). Selection uses only the training editors' val edits (gpt-oss, Gemma, later GLM 5.3 Flash, which
      becomes a training editor); Qwen3.5 val and test stay untouched until the final model. Val metrics are also
      reported per training editor, pass ("edit", "light") and source, so in-distribution detection is checked
      editor by editor before any held-out test.
    - Decided 2026-11-01 by Ege, isolated-sentence grid (20% of train documents): frozen embedders with a
      logistic-regression head (Qwen3-Embedding 0.6B/4B/8B, EmbeddingGemma-300m, Llama-Embed-Nemotron-8B);
      encoders fine-tuned end to end (DeBERTa-v3-large, ModernBERT-large, Ettin-encoder-1B) x 3 learning rates x
      2 weightings (global class weights vs per-sentence weights); decoders with LoRA and a classification head
      (Llama-3.1-8B, OLMo-3-7B, Ettin-decoder-1B) x 2 learning rates. A Qwen decoder as a family-bias arm was not
      chosen. Class imbalance: inverse-frequency class weights in the cross-entropy.
    - Decided 2026-11-01 by Ege, document detector (backbones ModernBERT-large, Ettin-encoder-1B, Llama-3.1-8B,
      started in parallel with the isolated grid):
      - Training versions: splits are by document and fixed first; versions are built from train documents only.
        Per document of n sentences: one all-human version, plus, for every training editor and pass, one version per
        theta in {0.1, ..., 0.9} with m = round(n^(1 - theta)) sentences replaced by that editor-pass's accepted
        edits (one editor-pass per version, as in the phase-diagram documents). Val documents get versions once, by
        the same recipe with a fixed seed, saved with their edit positions and reused by every run.
      - Every sentence of every version is a labelled example (edited 1, human 0), class-weighted.
      - One pass per version: encoders pool each sentence's tokens and label all sentences at once. Llama reads the
        chunk twice under its usual causal attention and each sentence is scored from its last token in the second
        copy, which has seen the whole chunk (echo; decided 2026-11-01 by Ege, replacing per-sentence probes behind a
        custom attention mask: same token cost, standard attention, a published method, Springer et al. 2024), so a
        sentence's score depends only on the document.
      - Long documents are cut at the end and the remainder becomes the next chunk, at sentence boundaries, so every
        sentence is labelled. Chunk length: 2,048 tokens (decided 2026-11-01 by Ege).
      - Caveats: each swapped-in edit was generated against the all-human document, so multi-edit versions are
        stitched from independent single-sentence edits (test documents are built the same way); a human sentence's
        score depends on its neighbors (item 27), so per-sentence p-values for recovery are not exactly uniform in
        edited documents, while the all-human calib documents still give a valid null for the global test.
    - Decided 2026-11-01 by Ege: Llama-3.1-8B is the base model (meta-llama/Llama-3.1-8B); the embedder is
      EmbeddingGemma 2 (google/embeddinggemma-2), not embeddinggemma-300m. Hyperparameters from a literature and
      model-card search (2026-11-01), all in `detector` in the config:
      - Embedders: each card's classification input (Qwen3 and Nemotron: "Instruct: Classify the given sentence as
        either human-written or edited by an AI model\nQuery:", our wording, as the cards ask for a task
        instruction; EmbeddingGemma 2: its "Classification" prompt), the model's own pooling, L2-normalized, 512
        tokens; standardized on train, then L2 logistic regression with balanced class weights, C in
        10^{-6,-4,...,6} chosen on val (CLIP linear-probe protocol; the interval refinement is not done).
      - Encoders: 2 epochs, effective batch 32, linear decay after 6% warmup, weight decay 0.01, gradient clipping
        1.0, 256 tokens, bf16 autocast with fp32 weights; DeBERTa-v3-large LRs 5e-6, 1e-5, 2e-5 (paper grid, card
        6e-6; 2e-5 is above the authors' grid), CLS pooling, AdamW eps 1e-6; ModernBERT-large 1e-5, 3e-5, 5e-5 (its
        GLUE picks), mean pooling, betas (0.9, 0.98), eps 1e-6; Ettin-encoder-1B 1e-5, 2e-5, 4e-5 (thin evidence:
        the paper reports only its sweep range), as ModernBERT.
      - Decoders: LoRA r 16, alpha 32, dropout 0.05 on all linear layers, head trained in full, last-token pooling,
        right padding; LRs 5e-5 and 2e-4 (QLoRA's 2e-4, LS-LLaMA's 8e-5; LoRA's best LR is about 10x full
        fine-tuning); otherwise as the encoders. OLMo-3-7B is the base model (allenai/Olmo-3-1025-7B), as for Llama.
      - Document detectors, one setting each (thin evidence: carried over from the sentence grid): 3 epochs,
        8 chunks per step, 10% warmup; ModernBERT-large 3e-5, Ettin-encoder-1B 2e-5, Llama-3.1-8B LoRA 1e-4 with
        gradient checkpointing, pooling each sentence's last token in the echo copy.
      - Checkpoint selection: 10 val evaluations per epoch (document detectors: 1), keeping the best TPR at 1% FPR;
        the same val then picks the search winner. Selection is on the training editors' val only.
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
    - Measured 2026-11-01 (Rui, [pilot-results.md](pilot-results.md)): intraclass correlation of human sentence z
      within test documents is 0.07-0.28 per source for the HC3 RoBERTa detector, and on human-only documents
      (n >= 16) Stouffer rejects 24%, HC+ 17%, HC 9%, min-p 6% at nominal 5%. Proposed fix there: thresholds under a
      Sigma estimated from a held-out half of calib.
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
    - Preliminary (2026-10-31, Rui): whole mode skips test documents under 16 sentences
      (`phase.whole_min_sentences`); the corpus itself still keeps 3-512 (item 12). Test-split lengths of the frozen
      corpus (sha256 8fba77d9): 2,914 documents, median 14 sentences, 3 to 479.

      | source | docs | median n | docs >= 16 | sentences in them | docs >= 32 |
      |---|---|---|---|---|---|
      | ellipse | 376 | 19 | 231 | 6,044 | 46 |
      | enron | 1,010 | 5 | 77 | 1,977 | 14 |
      | fineweb_edu | 218 | 23 | 149 | 7,101 | 83 |
      | openreview | 344 | 19 | 207 | 6,141 | 58 |
      | persuade | 349 | 20 | 241 | 6,309 | 51 |
      | pes2o | 38 | 190 | 38 | 7,642 | 38 |
      | writingprompts | 175 | 36 | 155 | 7,300 | 101 |
      | xsum | 404 | 16 | 203 | 5,458 | 52 |
      | all | 2,914 | 14 | 1,301 | 47,972 (79%) | 443 |

    - Review: 16 matches the smaller window and keeps at least 77 documents per source; 32 would leave Enron with
      14. A cutoff changes the source mix (Enron is mostly 3-8 sentences, peS2o all 41+), so report per source and
      per length band rather than pooled. peS2o has 38 test documents at any cutoff, so its intervals are wide.
      Monte Carlo thresholds are computed per distinct length (124 lengths >= 16, about 3 minutes at 100,000 reps).
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
    - Preliminary: one base seed (0), with per-cell seeds derived from it. Every per-cell mean carries a 95%
      interval over documents in `<metric>_lo` / `<metric>_hi` (`phase.ci_level`, `phase.summarize`): Wilson for
      rates (rejection, exact recovery, coverage), mean +- 1.96 sd / sqrt(N) for Hamming error and the
      edited-fraction columns. Documents are treated as independent, which holds for the draw (one window per
      document) but not for everything upstream.
    - Review: the intervals ignore (a) the shared calibration set behind every conformal p-value, (b) the
      empirical-null fit pooled over all z of a cell, (c) the Monte Carlo test thresholds, and (d) the choice of
      one rep per sentence. A bootstrap over documents that refits (b) would cover (b); (a) needs resampling
      the calibration split. Cells at different theta reuse the same documents, so their intervals are not
      independent; compare cells by paired differences, not by overlapping intervals.
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
    - Superseded 2026-11-04 by items 67 and 68: the corpus's own documents and much smaller models.
67. Pretraining data. Decided 2026-11-04 by Ege.
    - The detection corpus's own documents instead of C4. The models train on the train, val and calib splits
      (11,740 documents, 4.64M words, about 6.3M Pythia tokens by an estimated 1.35 tokens per word) and are evaluated
      by perplexity on the test split's human documents (2,914 documents, 1.10M words), held out from every arm.
    - Calib documents have no edits, so they stay human in every arm.
68. Model sizes. Decided 2026-11-04 by Ege.
    - Llama-style blocks (RoPE, SwiGLU, RMSNorm), tied embeddings, 1 epoch.
    - Chinchilla-optimal: d_model 64, 6 layers, about 0.3M non-embedding parameters (about 20 tokens per parameter
      on the human training tokens). Data-starved: d_model 128, 8 layers, about 1.6M (about 5x past Chinchilla).
69. Tokenizer. Decided 2026-11-04 by Ege: Pythia's (GPT-NeoX BPE, about 50k tokens). Llama 3's 128k vocabulary was
    set aside because its embedding table would be dozens of times the rest of models this small.
70. Arms and weights. Decided 2026-11-04 by Ege; every run at both sizes with 3 seeds, for error bars.
    - human: the human documents.
    - mixture (a mean over editors and sparsity): each train or val document once, with theta drawn uniformly from
      `phase.thetas`, one editor-wording drawn uniformly from the training editors' 10, and m = round(n^(1 - theta))
      random sentences replaced by that editor-wording's accepted edit (a position without one stays human), as in
      the phase-diagram documents.
    - all-edited ("squeeze as much information as possible from AI-edited text"): each train or val document once
      per training editor-wording (10 copies), every sentence replaced by that editor-wording's accepted edit where
      one exists; calib documents (no edits) are left out. With the mask weight only the sentences an editor-wording
      left without an accepted edit carry weight, with the AI-edited sentences as context.
    - Oracle weights, per sentence and applied to its tokens: mask, 0 for an edited sentence and 1 for a human one;
      strength, 1 - min(actual_r, 1), where actual_r is changed words over words. The loss is the weighted mean over
      tokens.
    - The held-out editor's (Qwen's) edits are not used. The seed sets the model initialization, the data order and
      the mixture draw.
    - Controls, decided 2026-11-05 by Ege after the first results: mixture-unweighted and all-edited-unweighted (the
      same documents with every weight 1; the mixture draw is the same for a given seed), and human-repeated (each
      train or val human document 10 times, calib left out: the documents, passes and tokens of all-edited without the
      edits), to separate the effect of the weights and of AI-edited text from that of more tokens.
    - Fresh-data control, decided 2026-11-05 by Ege: human-fresh trains on the train and val human documents plus new
      documents from the same sources (`fresh.py`: the documents after the corpus's in each seeded stream, through
      corpus.py's repairs, segmenter and filters), 9 times each source's train and val words where it has them, so
      it has about all-edited's tokens in distinct documents. Option (a): sources that run out (ELLIPSE and Enron,
      by an unverified estimate of their sizes) stay short rather than being filled from others, so results are
      compared per source and the pooled source mix shifts.
    - One-pass comparison, decided 2026-11-05 by Ege: all-edited with one editor-wording drawn per document
      (unweighted, strength and mask) against human-1pass (the same train and val documents, human), alongside the
      10-pass all-edited arms against human-repeated, to see how close AI-edited text gets to human text at each.
      "AI-edited" keeps the sentences whose edit was rejected human (about 6.5% of all-edited's tokens), which are
      the only ones the mask arms train on.
    - Not run yet: document-level weights from the to-do list.
71. Training settings. Preliminary (chosen by Claude 2026-11-04, not yet reviewed by Ege; `pretrain` in the config).
    - Documents tokenized sentence by sentence with their separators, shuffled, joined with EOS, cut into 512-token
      windows (plus one target), windows shuffled; 16 windows per step, 1 epoch.
    - AdamW, lr 2e-3, betas (0.9, 0.95), weight decay 0.1 on matrices, gradient clipping 1.0, 5% linear warmup then
      cosine decay to 10%; bf16 autocast.
    - Loss: weighted token cross-entropy divided by the step's tokens times the arm's mean weight, so each step's
      gradient scales with how much weight it carries (weighted risk over the corpus); EOS tokens get weight 1.
    - Test perplexity on the test split's human documents, pooled and per source, 10 times per run.

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

## Edited fraction (`stats.eps_storey`, `stats.eps_lower`)

The week-2 estimator target, taken here as the fraction eps = m / n of a document's sentences that were edited
(the rare/weak eps). Both estimators are row-wise functions of p-values, so they apply unchanged to the corpus-level
fraction of edited documents given one p-value per document (e.g. from HC).

67. Estimators. TBD.
    - Preliminary: Storey (2002) point estimate 1 - #{p > lambda} / ((1 - lambda) n), clipped to [0, 1], at
      lambda = 1/2 (`phase.storey_lambda`); Meinshausen & Rice (2006) lower confidence bound with the HC bounding
      function at level `phase.alpha`, whose constant is the Monte Carlo null quantile of HC over all i
      (`null_thresholds` key `hc_all`). The bound is positive exactly when that HC rejects. Reported per cell:
      `eps_true` (sentences actually edited, so rejected edits count as human), `eps_storey`,
      `sqerr_eps_storey`, `eps_lower`, `cover_eps_lower`.
    - Gaussian check (n in {16, 32}, the `sim_r` grid): MR coverage >= 0.95 in every cell, but the bound is
      conservative (n = 32, theta = 0.5, r = 2: eps = 0.19, mean bound 0.14; at r = 0.5 it is 0.03). Storey is
      close for dense strong edits, biased down by eps P(edited p > 1/2) / (1/2) for weak edits, and biased up by
      the clip at 0 for sparse ones (n = 32, theta = 0.9: eps = 0.031, mean estimate 0.08-0.09).
    - Review: whether per-document eps is the quantity the week-4 weights need, or the corpus-level fraction
      of edited documents (Lecture 5 lists "estimate fraction" at the document level); the choice of lambda;
      Jin & Cai (2007) for a consistent estimator in the sparse regime where Storey fails; a pooled estimate per
      cell (all windows together) in place of the mean of per-document estimates. Coverage on real p-values
      inherits the independence assumption of item 44.
