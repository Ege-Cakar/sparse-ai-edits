"""Sentence and paragraph breaks from an LLM at an OpenAI-compatible endpoint (corpus.segmenter), SaT as fallback.

A document's paragraphs (text.paragraphs) go to the model in chunks of whole paragraphs, joined by the source's
paragraph separator. The model replies with one sentence per line and an empty line between paragraphs. A reply is
accepted only if, with whitespace removed, it equals the chunk, except that it may drop a hyphen standing before
whitespace ("in- dicates"), or in segmenter.rejoin_hyphens sources between two lowercase letters ("pop-ular"), and
curly and straight quotes count as equal; otherwise, or if the reply was cut off, the chunk is split by SaT
(text.indexed_sentences). Sentences are always rebuilt from the chunk's own characters: in the sources of
segmenter.fix_spacing with the model's spacing and dropped hyphens, elsewhere with the original spacing. A paragraph
break is kept only where the chunk had a line break; a line break inside a model sentence is line wrapping. The model
sometimes leaves two sentences on one line, so SaT's breaks inside a model sentence are added where the text before
the break ends a sentence (resplit; those sentences are marked "llm+sat"). Replies are cached by request in
segment_cache.jsonl, which, not the server, defines the corpus: greedy replies to concurrent identical requests can
differ, so identical chunks in one batch share a request and a key written twice keeps its first reply.
"""
import asyncio
import json
import re
from itertools import accumulate

import openai

from .sources import text_key
from .text import _split

DESCRIPTION = {"fineweb_edu": "an educational web page", "pes2o": "a scientific paper extracted from PDF",
               "persuade": "a student essay", "ellipse": "an essay by an English-language learner",
               "xsum": "a BBC news article", "openreview": "a peer review of a research paper",
               "enron": "a business email", "writingprompts": "a short story posted on Reddit",
               # third-party datasets (thirdparty.py), keyed dataset/domain
               "editlens/amazon_reviews": "an Amazon product review",
               "editlens/google_reviews": "a Google review of a local business",
               "editlens/fineweb_edu": "an educational web page", "editlens/news": "a news article",
               "editlens/reddit_writing_prompts": "a short story posted on Reddit",
               "editlens/enron_email": "a business email",
               "apt_eval/blog": "a personal blog post", "apt_eval/email_content": "an email",
               "apt_eval/game_review": "a video game review", "apt_eval/news": "a news article",
               "apt_eval/paper_abstract": "the abstract of a scientific paper", "apt_eval/speech": "a speech transcript",
               "apt_eval/TruthfulQA_LLMs": "an answer to a question", "apt_eval/NarrativeQA_LLMs": "a story summary",
               "apt_eval/SQuAD1_LLMs": "an encyclopedia passage",
               "peer_review/peer_review": "a peer review of a research paper",
               "raid/abstracts": "the abstract of a scientific paper", "raid/books": "a plot summary of a book",
               "raid/news": "a news article", "raid/poetry": "a poem", "raid/recipes": "a recipe",
               "raid/reddit": "a Reddit post", "raid/reviews": "a movie review", "raid/wiki": "a Wikipedia article",
               "raid/unknown": "a text (an abstract, book summary, news article, poem, recipe, Reddit post, movie "
                               "review or Wikipedia article)"}
PROMPT = """Split this text, {description}, into sentences. Write each sentence on its own line, never two sentences \
on one line, with an empty line between paragraphs.
- A line break inside a sentence is line wrapping: join the lines.
- A heading, a signature line or a link is its own line; a list number stays with its item.
- {spacing}
- Copy every other character exactly, including quotation marks and apostrophes, spelling mistakes, punctuation and \
capitalization. Do not add, remove or reword anything.
Reply with the split text only.

Text:
{text}"""
FIX = ("The text was processed automatically and may have spacing errors: fix them, e.g. \"do n't\" -> \"don't\", "
       "\"( word )\" -> \"(word)\", \"CO 2\" -> \"CO2\", \"in- dicates\" -> \"indicates\". Change only spaces, and "
       "hyphens inside words broken across lines.")
KEEP = "Keep the spacing exactly as it is."
END = re.compile(r"[.!?…][\"'”’)\]]*$")   # a sentence end: . ! ? or …, then closing quotes or brackets
SAME = str.maketrans("‘’“”", "''\"\"")   # the model may straighten quotes; the original characters are kept


def chunks(paras, max_chars):
    """Consecutive paragraphs in chunks of at most max_chars characters; a longer paragraph is a chunk by itself."""
    out = [[]]
    for p in paras:
        if out[-1] and sum(map(len, out[-1])) + len(p) > max_chars:
            out.append([])
        out[-1].append(p)
    return out if out[0] else []


def droppable(src, j, inner_hyphens):
    """Whether the hyphen src[j] may be dropped: it stands before whitespace, or between lowercase letters."""
    after = src[j + 1:j + 2]
    return after.isspace() or inner_hyphens and src[j - 1:j].islower() and after.islower()


def align(src, out, inner_hyphens=False):
    """Index in src of each non-whitespace character of out, or None unless out equals src up to whitespace, hyphens
    that stand before whitespace in src (with inner_hyphens, also between two lowercase letters), and curly vs straight
    quotes."""
    idx, j, src_ = [], 0, src.translate(SAME)
    for c in out.translate(SAME):
        if c.isspace():
            continue
        while j < len(src) and (src[j].isspace() or src[j] == "-" != c and droppable(src, j, inner_hyphens)):
            j += 1
        if j == len(src) or src_[j] != c:
            return None
        idx.append(j)
        j += 1
    return idx if not src[j:].strip() else None


def from_reply(src, reply, fix_spacing, inner_hyphens=False):
    """[(starts_paragraph, sentence)] from a reply to the chunk src, or None if the reply changed more than allowed."""
    paras = [[s.strip() for s in b.splitlines() if s.strip()] for b in re.split(r"\n\s*\n", reply.strip())]
    flat = [(k == 0, s) for para in paras for k, s in enumerate(para)]
    idx = align(src, "".join(s for _, s in flat), inner_hyphens)
    if idx is None or not flat:
        return None
    ends = list(accumulate(sum(not c.isspace() for c in s) for _, s in flat))
    out, prev = [], 0
    for (new_para, s), a, b in zip(flat, [0] + ends[:-1], ends):
        lo, hi = idx[a], idx[b - 1] + 1
        pos = iter(idx[a:b])
        text = "".join(c if c.isspace() else src[next(pos)] for c in s) if fix_spacing else src[lo:hi]
        out.append((not out or new_para and "\n" in src[prev:lo], " ".join(text.split())))
        prev = hi
    return out


def resplit(sents):
    """[(starts_paragraph, sentence, segmenter)] with SaT's breaks added inside "llm" sentences where the text before
    the break ends a sentence; the cut is made in the sentence itself, so no character changes."""
    llm = [k for k, (_, _, how) in enumerate(sents) if how == "llm"]
    pieces = dict(zip(llm, (parts for parts, _ in _split([sents[k][1] for k in llm])))) if llm else {}
    out = []
    for k, (new, s, how) in enumerate(sents):
        cuts, pos = [], 0
        for part in pieces.get(k, []):
            pos = s.find(part, pos)
            cuts.append(pos)
            pos += len(part)
        cuts = [0] + [c for c in cuts[1:] if c > 0 and END.search(s[:c].rstrip())] if -1 not in cuts else [0]
        if len(cuts) == 1:
            out.append((new, s, how))
        else:
            ends = cuts[1:] + [len(s)]
            out += [(new and i == 0, s[a:b].strip(), "llm+sat") for i, (a, b) in enumerate(zip(cuts, ends))]
    return out


class Segmenter:
    """Splits batches of documents through the endpoint, with the reply cache at cache_path."""

    def __init__(self, cfg, cache_path):
        self.cfg, self.path, self.cache = cfg, cache_path, {}
        for r in map(json.loads, cache_path.read_text().splitlines() if cache_path.exists() else []):
            self.cache.setdefault(r["key"], r["reply"])

    def __call__(self, source, sep, docs):
        """For each document (a list of paragraphs): [(para_idx, sentence, segmenter)], segmenter "llm", "sat" or
        "punkt"."""
        return asyncio.run(self.batch(source, sep, docs))

    async def batch(self, source, sep, docs):
        client = openai.AsyncOpenAI(base_url=self.cfg["base_url"], api_key="EMPTY", max_retries=8, timeout=1800)
        sem, self.pending = asyncio.Semaphore(self.cfg["concurrency"]), {}
        with open(self.path, "a") as cache:
            done = await asyncio.gather(*[asyncio.gather(*[self.chunk(client, sem, cache, source, sep, c)
                                                         for c in chunks(paras, self.cfg["max_chunk_chars"])])
                                          for paras in docs])
        await client.close()
        out = []
        for doc in done:
            sents = resplit([x for part in doc for x in part])   # (starts_paragraph, sentence, segmenter)
            out.append([(p - 1, s, how) for p, (_, s, how) in zip(accumulate(int(new) for new, _, _ in sents), sents)])
        return out

    async def chunk(self, client, sem, cache, source, sep, paras):
        src = sep.join(paras)
        prompt = PROMPT.format(description=DESCRIPTION[source], text=src,
                               spacing=FIX if source in self.cfg["fix_spacing"] else KEEP)
        key = text_key(json.dumps([self.cfg["model"], prompt]))
        if key not in self.cache:
            if key not in self.pending:   # identical chunks in one batch share one request
                self.pending[key] = asyncio.ensure_future(self.ask(client, sem, cache, key, prompt, len(src)))
            await self.pending[key]
        sents = self.cache[key] and from_reply(src, self.cache[key], source in self.cfg["fix_spacing"],
                                               source in self.cfg["rejoin_hyphens"])
        if sents:
            return [(new, s, "llm") for new, s in sents]
        return [(i == 0, s, how) for ss, how in _split(paras) for i, s in enumerate(ss)]

    async def ask(self, client, sem, cache, key, prompt, n_chars):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=self.cfg["model"], messages=[{"role": "user", "content": prompt}], temperature=0,
                    max_tokens=n_chars // 2 + 256, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            reply = r.choices[0].message.content if r.choices[0].finish_reason == "stop" else None
        except openai.BadRequestError:   # e.g. a chunk longer than the served context
            reply = None
        self.cache[key] = reply
        cache.write(json.dumps({"key": key, "reply": reply}) + "\n")
        cache.flush()
