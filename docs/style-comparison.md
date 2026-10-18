# Style comparison study (2026-10-13)

Question: which cheaper editor makes edits most like the frontier editor's (`gpt-6-astra`)?

## Design

- 88 sentences (11 per source, 10-40 words) from `studies/style_comparison.yaml`; target strengths 0.1, 0.3, 0.5;
  2 samples per (sentence, strength) per editor; same prompt as the main pipeline; default sampling temperature.
- Editors through the HUIT gateway: `gpt-6-astra` and `gpt-6.1-sol` at `reasoning_effort: low` (their lowest
  setting), `gpt-6-luna` at `low` and at `none`.
- Astra's second sample was capped by budget, so the analysis uses the 142 (sentence, strength) units from 49
  sentences where every editor has 2 accepted edits (4-9 sentences per source).
- Metrics (`studies/style_comparison.py`): overlap of edit sites on the original sentence (Jaccard), word-level
  distance between two edited versions, and Jensen-Shannon divergence of introduced words. Each editor's
  self-similarity (its two samples compared with each other) is the ceiling for how close another editor can be.
  95% intervals bootstrap over sentences.

## Cost

| Editor | Cost per edit | Spent |
|---|---|---|
| `gpt-6-luna`, reasoning none | $0.00003 | $0.015 |
| `gpt-6-luna`, reasoning low | $0.00029 | $0.155 |
| `gpt-6.1-sol`, reasoning low | $0.0029 | $1.55 |
| `gpt-6-astra`, reasoning low | $0.0143 | $5.84 |

## Results

All four hit the requested strength (median measured strength 0.1 / 0.3 / 0.5; luna with reasoning off 0.47 at
0.5). Rejections were under 2%.

Similarity to astra, as the difference from astra's self-similarity (0 = as close as astra is to itself):

| Editor | Edit-site overlap | Output distance |
|---|---|---|
| `gpt-6.1-sol`, low | -0.026 [-0.072, 0.019] | +0.010 [-0.008, 0.029] |
| `gpt-6-luna`, low | -0.167 [-0.213, -0.122] | +0.093 [0.073, 0.113] |
| `gpt-6-luna`, none | -0.225 [-0.285, -0.167] | +0.122 [0.093, 0.149] |

Astra's self-similarity: overlap 0.643, distance 0.189. Introduced-word divergence (bits) from astra: sol 0.481,
luna low 0.561, luna none 0.614, against astra's own 0.449 between its two samples (no intervals computed).

Reading: on these surface measures, sol's edits are as close to astra's as astra's are to each other, at about a
fifth of the cost. The data cannot rule out a gap of up to 0.07 in site overlap. Luna is clearly different under
either reasoning setting, and closer with reasoning on. With reasoning off, luna is very consistent with itself
(self overlap 0.673) but consistent in its own way.

## Limits

- 49 sentences, one prompt, three strengths, default temperature. Too small for per-source or per-strength results.
- These metrics compare where and how text changes, not whether detectors respond the same way. The deciding test is
  detector transfer (a detector trained on one editor's edits, tested on another's), which needs the RunPod
  detectors.
- Two astra requests failed with gateway connection errors and were not retried; they are outside the 142 units.

## Claude models via subagents (2026-10-13)

Same 88 sentences, strengths and 2 samples, edited by Claude Code subagents (`opus`, `sonnet`, `haiku` aliases;
Opus / Sonnet / Haiku 5.5). Each subagent got one task file (one sample at one strength, 88 prompts built by the
pipeline's own prompt function), was told to answer each prompt as a separate conversation with the pipeline's system
message, and wrote its edits by hand. `agent-tasks` / `agent-collect` in `studies/style_comparison.py` build the task
files and apply the pipeline's cleaning and rejection rules (one attempt, no resampling). No subagent ran Bash; three
Haiku subagents also used Edit on their own output file. One Haiku subagent hit the 128k output-token limit and was
rerun with the same prompt. Cost: about 3M tokens of Claude Code usage, none of the HUIT quota.

All three hit the requested strength (medians 0.1 / 0.3 / 0.5); rejections 0.6-1.7%.

Within Claude (255 units, 86 sentences), difference from Opus's self-similarity (overlap 0.642, distance 0.180):

| Editor | Edit-site overlap | Output distance |
|---|---|---|
| Sonnet | -0.208 [-0.246, -0.166] | +0.100 [0.082, 0.118] |
| Haiku | -0.272 [-0.314, -0.231] | +0.136 [0.119, 0.155] |

All seven editors together (141 units, 49 sentences), edit-site overlap difference from astra's self-similarity:
sol -0.03, Sonnet -0.15, luna low -0.17, Haiku -0.21, luna none -0.23, Opus -0.26. From Opus's self-similarity,
every other editor is between -0.21 (Sonnet) and -0.31 (luna low).

Reading: unlike the OpenAI pair, neither smaller Claude model edits like Opus. Opus is the most self-consistent and
the most distinctive editor in the set; Sonnet's edits are closer to astra's than to Opus's.

Limits beyond those above: the Claude edits come from agents inside Claude Code (its system prompt, 88 sentences per
conversation, an instruction not to reuse wording across lines, unknown sampling settings), not isolated API calls,
so Claude-vs-OpenAI gaps mix model and setup. Comparisons among the three Claude models share one setup.
