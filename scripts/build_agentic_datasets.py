#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Build agentic task datasets from the three existing corpora.

Produces, for each task:

1. ``sft_tooluse.json``       - SFT warm-up trajectories that teach the base
                                SLM the tool-call syntax (needed so PPO can
                                get above the tool-call-accuracy threshold
                                of 0.5 predicted by the refined hypothesis).
2. ``task_prompts.json``      - evaluation / PPO training prompts with any
                                task metadata needed at rollout time
                                (constraints for A, entities for B, gold
                                answers for C).
3. ``trajectory_prefs.json``  - pairs of full trajectories (good vs broken),
                                used to train the trajectory-level Bradley
                                -Terry reward model.

Usage::

    python scripts/build_agentic_datasets.py \
        --task all \
        --input_dir ./data \
        --output_dir ./data/agentic \
        --num_train 2000

Assumes the original SFT splits produced by ``prepare_all_datasets.py`` are
already present in ``--input_dir``. Does not download anything from HF.
"""
# from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _split_paragraphs(text: str, min_len: int = 40) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paras if len(p) >= min_len]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _extract_entities_simple(text: str, k: int = 5) -> List[str]:
    """Heuristic capitalised-noun entity extractor. Good enough for Task B.

    Avoids spaCy as a hard dependency. The top-k capitalised tokens that are
    not sentence-initial are returned.
    """
    tokens = re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", text)
    # Drop common sentence-initial throwaways.
    blocklist = {"The", "This", "These", "That", "There", "When", "While", "After"}
    candidates = [t for t in tokens if t not in blocklist]
    counts: Dict[str, int] = defaultdict(int)
    for c in candidates:
        counts[c] += 1
    return [e for e, _ in sorted(counts.items(), key=lambda t: -t[1])[:k]]


# ===========================================================================
# Task A - Interactive Story Agent (TinyStories)
# ===========================================================================

_CHARACTER_POOL = [
    "rabbit", "fox", "scientist", "robot", "cat", "sailor", "baker",
    "fairy", "turtle", "ghost", "farmer", "mermaid", "dragon", "child",
]
_SETTING_POOL = [
    "the Arctic", "a bustling city", "a quiet forest", "outer space",
    "a deep ocean", "a sunny meadow", "a haunted house", "a tiny island",
    "an old library", "a distant galaxy",
]
_TWIST_POOL = [
    "a surprise", "a lesson about kindness", "a clever trick", "an unlikely friendship",
    "an invention", "a found object", "a journey home",
]


def _sample_story_constraints(rng: random.Random) -> Dict[str, Any]:
    n_chars = rng.choice([1, 2])
    return {
        "characters": rng.sample(_CHARACTER_POOL, n_chars),
        "setting": rng.choice(_SETTING_POOL),
        "twist": rng.choice(_TWIST_POOL),
        "max_words": rng.choice([100, 120, 150, 180]),
    }


def _format_story_prompt(c: Dict[str, Any]) -> str:
    chars = " and ".join(f"a {x}" for x in c["characters"])
    return (
        f"Write a short story featuring {chars}, set in {c['setting']}, "
        f"that ends with {c['twist']}. Keep it under {c['max_words']} words."
    )


def _evaluate_story_constraints(text: str, c: Dict[str, Any]) -> Dict[str, float]:
    lower = text.lower()
    chars_present = sum(1 for ch in c["characters"] if ch in lower) / max(1, len(c["characters"]))
    setting_present = 1.0 if any(w in lower for w in _tokenize(c["setting"])[-2:]) else 0.0
    word_count = len(_tokenize(text))
    length_ok = 1.0 if word_count <= c["max_words"] else 0.0
    return {
        "characters": chars_present,
        "setting": setting_present,
        "length": length_ok,
        "total": (chars_present + setting_present + length_ok) / 3.0,
    }


def _synthetic_story_tooluse_demo(prompt: str, constraints: Dict[str, Any], gold: str) -> str:
    """A one-shot SFT demonstration that teaches tool-call syntax on Task A."""
    chars_json = json.dumps(constraints["characters"])
    return (
        f"{prompt}\n\n"
        f"<tool name=\"length_check\">"
        f"{{\"text\": \"(draft omitted)\", \"max_words\": {constraints['max_words']}}}"
        f"</tool>\n"
        f"<r>word_count=0 max_words={constraints['max_words']} within_budget=true</r>\n"
        f"Thinking: I will draft a short story that uses all required elements.\n"
        f"<tool name=\"character_check\">"
        f"{{\"text\": \"{gold[:200].replace(chr(34), chr(39))}\", \"characters\": {chars_json}}}"
        f"</tool>\n"
        f"<r>all_present=true missing=[]</r>\n"
        f"<finish>{gold}</finish>"
    )


def build_task_a(
    input_dir: Path,
    output_dir: Path,
    num_train: int,
    seed: int = 42,
) -> None:
    logger.info("[Task A] building from TinyStories")
    rng = random.Random(seed)
    raw = _read_json(input_dir / "tinystories" / "sft_train.json")
    rng.shuffle(raw)
    raw = raw[: max(num_train * 2, 500)]

    prompts: List[Dict[str, Any]] = []
    tooluse_sft: List[Dict[str, Any]] = []
    pref_pairs: List[Dict[str, Any]] = []

    for i, ex in enumerate(raw[:num_train]):
        story = ex.get("text", "").strip()
        if len(story) < 30:
            continue
        c = _sample_story_constraints(rng)
        task_prompt = _format_story_prompt(c)

        # Prompts for PPO / eval.
        prompts.append({
            "task_id": f"tinystories_a_{i}",
            "task": "story",
            "task_prompt": task_prompt,
            "constraints": c,
            "reference_story": story,
        })

        # Tool-use SFT demonstration using the gold story.
        tooluse_sft.append({
            "task_id": f"tinystories_a_{i}",
            "text": _synthetic_story_tooluse_demo(task_prompt, c, story),
        })

        # Build a preference pair: gold story vs a version that misses a character.
        if len(c["characters"]) >= 1:
            broken = story.replace(c["characters"][0], "someone")
            pref_pairs.append({
                "task_id": f"tinystories_a_{i}",
                "prompt": task_prompt,
                "chosen": story,
                "rejected": broken,
                "reason": f"missing_character:{c['characters'][0]}",
            })

    _write_json(output_dir / "task_a_tinystories" / "task_prompts.json", prompts)
    _write_json(output_dir / "task_a_tinystories" / "sft_tooluse.json", tooluse_sft)
    _write_json(output_dir / "task_a_tinystories" / "trajectory_prefs.json", pref_pairs)
    logger.info(
        f"[Task A] wrote {len(prompts)} prompts, {len(tooluse_sft)} SFT demos, "
        f"{len(pref_pairs)} preference pairs"
    )


# ===========================================================================
# Task B - Grounded Summarisation Agent (CNN/DailyMail)
# ===========================================================================

def _cnn_to_paragraphs(article: str) -> List[str]:
    # পূর্বে: শুধু `if not paras` চেক ছিল — কিন্তু CNN/DailyMail-এর `text`
    # ফিল্ডে newline প্রায় নেই, ফলে `_split_paragraphs` সবসময় ১টা প্যারা
    # ফেরত দিত এবং পরের `len(paragraphs) < 2` চেক সবগুলো রো বাদ দিত।
    # তাই ১টার বেশি প্যারা না পেলে sentence-chunking-এ ফ্যালব্যাক করা হলো।
    paras = _split_paragraphs(article)
    if len(paras) >= 2:
        return paras
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", article) if s.strip()]
    if len(sentences) < 2:
        return paras
    chunks = [" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]
    return [c for c in chunks if c]


def _synthetic_summary_tooluse_demo(
    prompt: str,
    paragraphs: List[str],
    reference: str,
    entities: List[str],
) -> str:
    top = paragraphs[:2]
    ents = json.dumps(entities)
    summary_line = reference.strip().split(". ")[0] + "."
    return (
        f"{prompt}\n\n"
        f"<tool name=\"retrieve\">{{\"query\": \"key facts\", \"k\": 2}}</tool>\n"
        f"<r>[0] {top[0][:120]}</r>\n"
        f"<tool name=\"cite_check\">"
        f"{{\"sentence\": \"{summary_line[:120]}\", \"paragraph_id\": 0}}</tool>\n"
        f"<r>grounded=true overlap_f1=0.4</r>\n"
        f"<tool name=\"coverage_check\">"
        f"{{\"summary\": \"{reference[:200]}\", \"reference_entities\": {ents}}}"
        f"</tool>\n"
        f"<r>covered=1.0</r>\n"
        f"<finish>{reference.strip()}</finish>"
    )


def build_task_b(
    input_dir: Path,
    output_dir: Path,
    num_train: int,
    seed: int = 42,
) -> None:
    logger.info("[Task B] building from CNN/DailyMail")
    rng = random.Random(seed)
    raw = _read_json(input_dir / "cnn_dailymail" / "sft_train.json")
    rng.shuffle(raw)

    prompts: List[Dict[str, Any]] = []
    tooluse_sft: List[Dict[str, Any]] = []
    pref_pairs: List[Dict[str, Any]] = []

    count = 0
    for i, ex in enumerate(raw):
        if count >= num_train:
            break
        text_field = ex.get("text", "")
        if "Article:" not in text_field or "Summary:" not in text_field:
            continue
        article = text_field.split("Article:", 1)[1].split("Summary:", 1)[0].strip()
        summary = text_field.split("Summary:", 1)[1].strip()
        paragraphs = _cnn_to_paragraphs(article)
        if len(paragraphs) < 2 or len(summary) < 30:
            continue

        entities = _extract_entities_simple(article + " " + summary, k=5)
        if not entities:
            continue

        task_prompt = (
            f"Write a 2-3 sentence grounded summary of the article titled "
            f"\"{article[:80]}...\". Cite at least two paragraphs."
        )
        prompts.append({
            "task_id": f"cnn_b_{i}",
            "task": "grounded_summary",
            "task_prompt": task_prompt,
            "article_paragraphs": paragraphs,
            "reference_summary": summary,
            "reference_entities": entities,
        })

        tooluse_sft.append({
            "task_id": f"cnn_b_{i}",
            "text": _synthetic_summary_tooluse_demo(
                task_prompt, paragraphs, summary, entities
            ),
        })

        # Preference pair: grounded reference vs ungrounded hallucination
        # (we reorder the entity list to create a "wrong emphasis" summary).
        broken = " ".join(reversed(summary.split(". ")))
        pref_pairs.append({
            "task_id": f"cnn_b_{i}",
            "prompt": task_prompt,
            "chosen": summary,
            "rejected": broken,
            "reason": "reordered_emphasis_loses_grounding",
        })
        count += 1

    _write_json(output_dir / "task_b_cnn" / "task_prompts.json", prompts)
    _write_json(output_dir / "task_b_cnn" / "sft_tooluse.json", tooluse_sft)
    _write_json(output_dir / "task_b_cnn" / "trajectory_prefs.json", pref_pairs)
    logger.info(
        f"[Task B] wrote {len(prompts)} prompts, {len(tooluse_sft)} SFT demos, "
        f"{len(pref_pairs)} preference pairs"
    )


# ===========================================================================
# Task C - Multi-hop Knowledge Agent (Wikitext-103)
# ===========================================================================

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

# পূর্বে: শুধু `was born in` ও `was founded in` দুইটা কঠিন প্যাটার্ন ছিল —
# Wikitext-103-এ এ ধরনের বাক্যগঠন বিরল, তাই একটা প্যারাগ্রাফেও দুটি ভিন্ন
# fact পাওয়া যেত না, ফলে multi-hop প্রশ্ন তৈরি হতো না (output 0)।
# এখন থেকে: একটা সাধারণ `<Entity> <verb> ... <year>` প্যাটার্ন +
# parenthetical birth-year প্যাটার্ন (e.g. "Alan Turing (1912 – 1954)")
# ব্যবহার করা হচ্ছে — দুটোই Wikitext গদ্যে অহরহ দেখা যায়।
_RELATION_VERBS: Dict[str, List[str]] = {
    "born_in":     ["was born in", "born in"],
    "died_in":     ["died in", "passed away in"],
    "founded_in":  ["was founded in", "was established in", "was created in"],
    "released_in": ["was released in", "was published in", "premiered in", "debuted in"],
    "located_in":  ["is located in", "is situated in", "is based in"],
    "written_by":  ["was written by", "was composed by", "was directed by", "was produced by"],
}

# Pre-compile each verb phrase as a regex that captures (entity, value).
# Entity is 1-3 capitalised tokens; value is a year, a capitalised noun
# phrase, or a single lowercased token (city / country names usually fall
# in the first two cases).
_RELATION_RES: List[Tuple[str, "re.Pattern[str]"]] = []
for rel, verbs in _RELATION_VERBS.items():
    for verb in verbs:
        verb_re = verb.replace(" ", r"\s+")
        pat = re.compile(
            r"\b(?P<ent>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\s+"
            + verb_re
            + r"\s+(?P<val>(?:1[5-9]\d{2}|20\d{2})|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})"
        )
        _RELATION_RES.append((rel, pat))

# "Alan Turing ( 1912 – 1954 )" style birth/death years.
_PAREN_LIFETIME_RE = re.compile(
    r"\b(?P<ent>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,2})\s*\(\s*"
    r"(?P<born>1[5-9]\d{2}|20\d{2})\s*[–\-]\s*(?P<died>1[5-9]\d{2}|20\d{2})\s*\)"
)

# "Entity ( 1990 )" — a single year next to a capitalised noun phrase
# (Wikitext-103 uses this for film / album / book release dates).
_PAREN_YEAR_RE = re.compile(
    r"\b(?P<ent>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\s*\(\s*"
    r"(?P<val>1[5-9]\d{2}|20\d{2})\s*\)"
)

# "In YYYY , Entity verbed ..." — encyclopaedic prose biographies use this
# almost universally. Captures (entity, occurred_in, year).
_IN_YEAR_RE = re.compile(
    r"\bIn\s+(?P<val>1[5-9]\d{2}|20\d{2})\s*,?\s+"
    r"(?P<ent>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\s+(?:was|is|became|joined|founded|wrote|released|directed|composed|published|moved|travelled|won|created)"
)


def _extract_facts(paragraphs: List[str]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    seen: set = set()  # de-dupe (entity, relation, value) triples per article
    for para_id, p in enumerate(paragraphs):
        for rel, pat in _RELATION_RES:
            for m in pat.finditer(p):
                key = (m.group("ent"), rel, m.group("val"))
                if key in seen:
                    continue
                seen.add(key)
                facts.append({
                    "entity": m.group("ent"),
                    "relation": rel,
                    "value": m.group("val"),
                    "paragraph_id": para_id,
                })
        for m in _PAREN_LIFETIME_RE.finditer(p):
            ent = m.group("ent")
            for rel, val in (("born_in", m.group("born")), ("died_in", m.group("died"))):
                key = (ent, rel, val)
                if key in seen:
                    continue
                seen.add(key)
                facts.append({
                    "entity": ent,
                    "relation": rel,
                    "value": val,
                    "paragraph_id": para_id,
                })
        for m in _PAREN_YEAR_RE.finditer(p):
            key = (m.group("ent"), "released_in", m.group("val"))
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "entity": m.group("ent"),
                "relation": "released_in",
                "value": m.group("val"),
                "paragraph_id": para_id,
            })
        for m in _IN_YEAR_RE.finditer(p):
            key = (m.group("ent"), "occurred_in", m.group("val"))
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "entity": m.group("ent"),
                "relation": "occurred_in",
                "value": m.group("val"),
                "paragraph_id": para_id,
            })
    return facts


def _make_multihop_question(fact_a: Dict[str, Any], fact_b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a two-hop question when two facts share an entity."""
    if fact_a["entity"] != fact_b["entity"]:
        return None
    if fact_a["relation"] == fact_b["relation"]:
        return None
    # e.g. "In what year was X born, and where were they founded?"
    ent = fact_a["entity"]
    question = (
        f"Using the passage, state two facts about {ent}: "
        f"{fact_a['relation'].replace('_', ' ')} and "
        f"{fact_b['relation'].replace('_', ' ')}. Give both values."
    )
    gold = {
        fact_a["relation"]: fact_a["value"],
        fact_b["relation"]: fact_b["value"],
    }
    return {"question": question, "gold": gold, "entity": ent,
            "evidence_paragraphs": [fact_a["paragraph_id"], fact_b["paragraph_id"]]}


def _synthetic_kb_tooluse_demo(
    question: str,
    paragraphs: List[str],
    gold: Dict[str, str],
    entity: str,
    evidence_ids: List[int],
) -> str:
    first_para = paragraphs[evidence_ids[0]][:150] if evidence_ids else ""
    answer = ", ".join(f"{k}={v}" for k, v in gold.items())
    return (
        f"{question}\n\n"
        f"<tool name=\"search_passage\">{{\"query\": \"{entity}\", \"k\": 2}}</tool>\n"
        f"<r>[{evidence_ids[0] if evidence_ids else 0}] {first_para}</r>\n"
        f"<tool name=\"entity_lookup\">{{\"entity\": \"{entity}\"}}</tool>\n"
        f"<r>(facts listed)</r>\n"
        f"<tool name=\"verify_fact\">"
        f"{{\"claim\": \"{answer[:100]}\", \"paragraph_id\": {evidence_ids[0] if evidence_ids else 0}}}"
        f"</tool>\n"
        f"<r>entailed=true overlap_f1=0.4</r>\n"
        f"<finish>{answer}</finish>"
    )


def build_task_c(
    input_dir: Path,
    output_dir: Path,
    num_train: int,
    seed: int = 42,
) -> None:
    logger.info("[Task C] building from Wikitext-103")
    rng = random.Random(seed)
    raw = _read_json(input_dir / "wikitext" / "sft_train.json")
    rng.shuffle(raw)

    # Group paragraphs into pseudo-articles of 5 paragraphs each.
    paragraphs = [ex.get("text", "").strip() for ex in raw if ex.get("text")]
    paragraphs = [p for p in paragraphs if len(p) > 80]

    prompts: List[Dict[str, Any]] = []
    tooluse_sft: List[Dict[str, Any]] = []
    pref_pairs: List[Dict[str, Any]] = []

    # পূর্বে: ৫-প্যারার ছোট window + প্রতিটা entity-র জন্য মাত্র ১টা প্রশ্ন
    # বানানো হতো, ফলে output ছিল মাত্র ৪০-এর কাছাকাছি। এখন window বাড়ানো
    # হয়েছে এবং একই entity-র যেকোনো দুই ভিন্ন (relation,value) জোড়াকে
    # multi-hop প্রশ্ন বানানোর জন্য ব্যবহার করা হচ্ছে।
    art_id = 0
    for start in range(0, len(paragraphs), 10):
        article_paras = paragraphs[start: start + 10]
        if len(article_paras) < 3:
            continue

        facts = _extract_facts(article_paras)
        if len(facts) < 2:
            continue

        by_entity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for f in facts:
            by_entity[f["entity"]].append(f)

        for ent, ent_facts in by_entity.items():
            if len(ent_facts) < 2 or len(prompts) >= num_train:
                continue
            # Try every distinct (i, j) pair of this entity's facts; emit the
            # first one that yields a valid multi-hop question (i.e. the two
            # facts use different relations).
            q = None
            for i in range(len(ent_facts)):
                for j in range(i + 1, len(ent_facts)):
                    candidate = _make_multihop_question(ent_facts[i], ent_facts[j])
                    if candidate is not None:
                        q = candidate
                        break
                if q is not None:
                    break
            if q is None:
                continue

            task_prompt = q["question"]
            prompts.append({
                "task_id": f"wikitext_c_{art_id}_{ent}",
                "task": "multi_hop_qa",
                "task_prompt": task_prompt,
                "article_paragraphs": article_paras,
                "gold_answer": q["gold"],
                "evidence_paragraphs": q["evidence_paragraphs"],
                "entity": ent,
            })

            tooluse_sft.append({
                "task_id": f"wikitext_c_{art_id}_{ent}",
                "text": _synthetic_kb_tooluse_demo(
                    task_prompt, article_paras, q["gold"], ent, q["evidence_paragraphs"]
                ),
            })

            # Preference pair: correct answer vs value-swapped incorrect.
            gold_str = ", ".join(f"{k}={v}" for k, v in q["gold"].items())
            wrong_val = "unknown"
            wrong_str = ", ".join(f"{k}={wrong_val}" for k in q["gold"])
            pref_pairs.append({
                "task_id": f"wikitext_c_{art_id}_{ent}",
                "prompt": task_prompt,
                "chosen": gold_str,
                "rejected": wrong_str,
                "reason": "wrong_value",
            })
            if len(prompts) >= num_train:
                break
        art_id += 1
        if len(prompts) >= num_train:
            break

    _write_json(output_dir / "task_c_wikitext" / "task_prompts.json", prompts)
    _write_json(output_dir / "task_c_wikitext" / "sft_tooluse.json", tooluse_sft)
    _write_json(output_dir / "task_c_wikitext" / "trajectory_prefs.json", pref_pairs)
    logger.info(
        f"[Task C] wrote {len(prompts)} prompts, {len(tooluse_sft)} SFT demos, "
        f"{len(pref_pairs)} preference pairs"
    )


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--input_dir", type=Path, default=Path("./data"))
    ap.add_argument("--output_dir", type=Path, default=Path("./data/agentic"))
    ap.add_argument("--num_train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task in ("a", "all"):
        build_task_a(args.input_dir, args.output_dir, args.num_train, args.seed)
    if args.task in ("b", "all"):
        build_task_b(args.input_dir, args.output_dir, args.num_train, args.seed)
    if args.task in ("c", "all"):
        build_task_c(args.input_dir, args.output_dir, args.num_train, args.seed)
    logger.info("done")


if __name__ == "__main__":
    main()

'''
python scripts/build_agentic_datasets.py --task all --num_train 2000
'''