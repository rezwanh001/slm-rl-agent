#-*- coding: utf-8 -*-
"""
Tool library for SLM-RL-Agents.

Each tool is a deterministic Python function of ``(args_dict, state) -> str``.
The environment dispatches to these by name; the base SLM never executes code.

The registry is split by task so you can build per-task tool subsets:

    tools_A = build_story_tools()
    tools_B = build_summarisation_tools(article_index)
    tools_C = build_kb_tools(kb, search_index)

All tools are pure (or read-only on ``state``) and safe to call in tight loops.

Integration: this module has no heavy dependencies beyond ``numpy`` and
``re``. ``sentence-transformers`` is imported lazily inside ``_build_sbert``
so the tests run without it.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ToolFn = Callable[[Dict[str, Any], Any], str]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _f1(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    common: Dict[str, int] = {}
    for w in ta:
        common[w] = min(ta.count(w), tb.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


def _flesch_kincaid_grade(text: str) -> float:
    """Rough, dependency-free FK grade level. Good enough for reward shaping."""
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = _tokenize(text)
    if not words:
        return 0.0
    syllables = 0
    for w in words:
        syllables += max(1, len(re.findall(r"[aeiouy]+", w)))
    return (
        0.39 * (len(words) / sentences)
        + 11.8 * (syllables / len(words))
        - 15.59
    )


# ---------------------------------------------------------------------------
# Task A: Interactive Story Agent (TinyStories)
# ---------------------------------------------------------------------------

def tool_length_check(args: Dict[str, Any], state: Any) -> str:
    text = args.get("text", "")
    budget = int(args.get("max_words", 150))
    n = len(_tokenize(text))
    return (
        f"word_count={n} max_words={budget} "
        f"within_budget={str(n <= budget).lower()}"
    )


def tool_character_check(args: Dict[str, Any], state: Any) -> str:
    text = args.get("text", "").lower()
    required = [c.lower() for c in args.get("characters", [])]
    found = {c: (c in text) for c in required}
    missing = [c for c, ok in found.items() if not ok]
    return (
        f"all_present={str(not missing).lower()} "
        f"missing={missing}"
    )


def tool_readability(args: Dict[str, Any], state: Any) -> str:
    grade = _flesch_kincaid_grade(args.get("text", ""))
    return f"fk_grade={grade:.1f}"


def build_story_tools() -> Dict[str, ToolFn]:
    return {
        "length_check": tool_length_check,
        "character_check": tool_character_check,
        "readability": tool_readability,
    }


# ---------------------------------------------------------------------------
# Task B: Grounded Summarisation Agent (CNN/DailyMail)
# ---------------------------------------------------------------------------

@dataclass
class ArticleIndex:
    """Per-article retrieval index with bag-of-words fallback.

    Sentence-BERT is used if available; otherwise BM25-style scoring is used.
    In practice the BoW fallback is sufficient for CNN/DailyMail paragraphs,
    which are short and entity-dense.
    """
    paragraphs: List[str]
    article_id: str
    _embeds: Optional[Any] = None
    _encoder: Optional[Any] = None

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, str, float]]:
        if self._encoder is not None and self._embeds is not None:
            return self._retrieve_sbert(query, k)
        return self._retrieve_bow(query, k)

    def _retrieve_bow(self, query: str, k: int) -> List[Tuple[int, str, float]]:
        q = set(_tokenize(query))
        scored: List[Tuple[int, str, float]] = []
        for i, p in enumerate(self.paragraphs):
            toks = set(_tokenize(p))
            if not toks:
                continue
            overlap = len(q & toks)
            score = overlap / math.sqrt(len(toks))
            scored.append((i, p, score))
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:k]

    def _retrieve_sbert(self, query: str, k: int) -> List[Tuple[int, str, float]]:
        import numpy as np
        q_emb = self._encoder.encode([query], normalize_embeddings=True)[0]
        sims = self._embeds @ q_emb
        idx = np.argsort(-sims)[:k]
        return [(int(i), self.paragraphs[int(i)], float(sims[int(i)])) for i in idx]


def _build_sbert_index(paragraphs: List[str], article_id: str) -> ArticleIndex:
    """Build an SBERT-backed index if the library is installed.

    Falls back to a pure-Python index if not. Either way the tool interface
    is identical, so the rest of the pipeline does not change.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeds = encoder.encode(paragraphs, normalize_embeddings=True)
        return ArticleIndex(
            paragraphs=paragraphs,
            article_id=article_id,
            _embeds=np.asarray(embeds),
            _encoder=encoder,
        )
    except ImportError:
        logger.info("sentence-transformers not installed; using BoW retrieval")
        return ArticleIndex(paragraphs=paragraphs, article_id=article_id)


def build_summarisation_tools(index: ArticleIndex) -> Dict[str, ToolFn]:
    """Build the retrieval+grounding toolset for one article."""

    def tool_retrieve(args: Dict[str, Any], state: Any) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 3))
        hits = index.retrieve(query, k=k)
        return "\n".join(
            f"[{i}] ({s:.2f}) {p[:300]}" for i, p, s in hits
        )

    def tool_cite_check(args: Dict[str, Any], state: Any) -> str:
        sentence = args.get("sentence", "")
        cite_id = int(args.get("paragraph_id", -1))
        if not 0 <= cite_id < len(index.paragraphs):
            return "grounded=false overlap_f1=0.0 reason=out_of_range"
        ref = index.paragraphs[cite_id]
        f1 = _f1(sentence, ref)
        grounded = f1 >= 0.25
        return f"grounded={str(grounded).lower()} overlap_f1={f1:.3f}"

    def tool_coverage_check(args: Dict[str, Any], state: Any) -> str:
        summary = args.get("summary", "")
        entities = args.get("reference_entities", [])
        if not entities:
            return "covered=1.0 reason=no_entities"
        hit = sum(1 for e in entities if e.lower() in summary.lower())
        return f"covered={hit/len(entities):.3f} hits={hit}/{len(entities)}"

    return {
        "retrieve": tool_retrieve,
        "cite_check": tool_cite_check,
        "coverage_check": tool_coverage_check,
        "length_check": tool_length_check,
    }


# ---------------------------------------------------------------------------
# Task C: Multi-hop Knowledge Agent (Wikitext)
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeBase:
    """A tiny KB of (entity, relation, value, evidence_paragraph) facts.

    Built offline by ``build_agentic_datasets.py`` from Wikitext paragraphs.
    """
    facts: List[Dict[str, str]]

    def lookup(self, entity: str, relation: Optional[str] = None) -> List[Dict[str, str]]:
        hits = []
        for f in self.facts:
            if f.get("entity", "").lower() == entity.lower():
                if relation is None or f.get("relation", "") == relation:
                    hits.append(f)
        return hits


def build_kb_tools(kb: KnowledgeBase, index: ArticleIndex) -> Dict[str, ToolFn]:

    def tool_search_passage(args: Dict[str, Any], state: Any) -> str:
        query = args.get("query", "")
        k = int(args.get("k", 3))
        hits = index.retrieve(query, k=k)
        return "\n".join(f"[{i}] {p[:300]}" for i, p, _ in hits)

    def tool_entity_lookup(args: Dict[str, Any], state: Any) -> str:
        entity = args.get("entity", "")
        relation = args.get("relation", None)
        hits = kb.lookup(entity, relation=relation)
        if not hits:
            return f"no_facts_for entity={entity!r} relation={relation!r}"
        return "\n".join(
            f"{h.get('entity','')} | {h.get('relation','')} | {h.get('value','')}"
            for h in hits[:5]
        )

    def tool_arithmetic(args: Dict[str, Any], state: Any) -> str:
        expr = args.get("expr", "").strip()
        if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
            return "error: only digits and + - * / ( ) allowed"
        try:
            # Evaluated in a locked-down namespace.
            value = eval(expr, {"__builtins__": {}}, {})
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"
        return f"result={value}"

    def tool_verify_fact(args: Dict[str, Any], state: Any) -> str:
        claim = args.get("claim", "")
        evidence = args.get("evidence", "")
        if not evidence:
            para_id = args.get("paragraph_id", None)
            if para_id is not None and 0 <= int(para_id) < len(index.paragraphs):
                evidence = index.paragraphs[int(para_id)]
        f1 = _f1(claim, evidence)
        entailed = f1 >= 0.3
        return f"entailed={str(entailed).lower()} overlap_f1={f1:.3f}"

    return {
        "search_passage": tool_search_passage,
        "entity_lookup": tool_entity_lookup,
        "arithmetic": tool_arithmetic,
        "verify_fact": tool_verify_fact,
    }


# ---------------------------------------------------------------------------
# Tool-use diagnostics (used by the paper's updated capacity-headroom check)
# ---------------------------------------------------------------------------

def tool_call_accuracy(raw_outputs: List[str], tools: Dict[str, ToolFn]) -> float:
    """Fraction of model outputs containing a syntactically valid tool call.

    Used in Week 1 to measure the SFT prior's tool-use competence before PPO.
    Call this on 50 prompts formatted with a tool-use few-shot prefix; a
    result below 0.5 predicts that PPO will not converge on that task.
    """
    from src.agentic.environment import parse_action, ActionType

    if not raw_outputs:
        return 0.0
    ok = 0
    for out in raw_outputs:
        a = parse_action(out)
        if a.type == ActionType.TOOL_CALL and a.tool_name in tools:
            ok += 1
    return ok / len(raw_outputs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Story tools
    tools = build_story_tools()
    print("length_check:", tools["length_check"]({"text": "one two three", "max_words": 5}, None))
    print("readability:", tools["readability"]({"text": "The cat sat. The dog ran."}, None))

    # Summarisation tools with a tiny article
    idx = ArticleIndex(
        paragraphs=[
            "Scientists announced a new telescope will launch in 2026.",
            "The project is funded by an international consortium.",
            "Astronomers expect sharper images of exoplanets.",
        ],
        article_id="demo",
    )
    s_tools = build_summarisation_tools(idx)
    print("retrieve:", s_tools["retrieve"]({"query": "telescope launch", "k": 2}, None))
    print("cite_check:", s_tools["cite_check"](
        {"sentence": "A telescope launches in 2026.", "paragraph_id": 0}, None
    ))

    # KB tools
    kb = KnowledgeBase(facts=[
        {"entity": "Alan Turing", "relation": "birth_year", "value": "1912"},
        {"entity": "Alan Turing", "relation": "death_year", "value": "1954"},
    ])
    kb_tools = build_kb_tools(kb, idx)
    print("entity_lookup:", kb_tools["entity_lookup"](
        {"entity": "Alan Turing"}, None
    ))
    print("arithmetic:", kb_tools["arithmetic"]({"expr": "1954 - 1912"}, None))
