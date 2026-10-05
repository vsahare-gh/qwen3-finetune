#!/usr/bin/env python3
"""Build a source-backed 80/20 QA fine-tuning set from local 3GPP text files.

The train/test JSONL files contain chat messages only. The QA evaluation prompts
contain questions only. Gold answers and evidence are kept in a separate file so
they can be reviewed without being shown to the model during inference.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "3gpp_specs" / "23-series-text"
DATA_DIR = ROOT / "data"
SEED = 42
TOTAL = 5_000
TRAIN_COUNT = 4_000

HEADING_RE = re.compile(
    r"^#{2,5}\s+(?:(?:\*\*)?(?P<clause>(?:\d+(?:\.[0-9A-Za-z]+)*|[A-Z](?:\.\d+)+))\s+)?(?P<title>.+?)\s*$"
)
GENERIC_RE = re.compile(
    r"^(general|introduction|overview|scope|references?|normative references|informative references|"
    r"definitions(?: and abbreviations)?|abbreviations|void|foreword|contents|history|change history|annex [A-Z])$",
    re.IGNORECASE,
)


def clean_title(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\*+|_+", "", value)
    value = re.sub(r"^[A-Z]\.\d+(?:\.\d+)*\s+", "", value)
    value = re.sub(r"^\d+(?:\.\d+)*\s+", "", value)
    return re.sub(r"\s+", " ", value).strip(" .:;—–-")


def clean_paragraph(raw_lines: list[str]) -> str:
    joined = " ".join(raw_lines)
    joined = re.sub(r"<!--.*?-->", " ", joined)
    joined = re.sub(r"<br\s*/?>", " ", joined, flags=re.IGNORECASE)
    joined = re.sub(r"<[^>]+>", " ", joined)
    joined = re.sub(r"!\[[^]]*\]\([^)]*\)", " ", joined)
    joined = re.sub(r"\*{1,3}|_{1,3}|`", "", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def extract_answer(section_lines: list[str]) -> str:
    # Prefer the first substantive prose paragraph or list block in the clause.
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in section_lines:
        s = line.strip()
        if not s:
            if current:
                blocks.append(current)
                current = []
            continue
        if s.startswith("|") or s.startswith(("![", "<!--", "**Figure", "Figure ")):
            continue
        s = re.sub(r"^[-*+]\s+", "", s)
        s = re.sub(r"^>\s*", "", s)
        if re.fullmatch(r"[0-9]{1,4}", s):
            continue
        if re.match(r"^(?:ETSI|3GPP TS|3GPP TR|Release [0-9]|Version [0-9])\b", s, re.IGNORECASE):
            continue
        if "version" in s.lower() and ("release" in s.lower() or "etsi" in s.lower()) and len(s) < 180:
            continue
        current.append(s)
    if current:
        blocks.append(current)

    for block in blocks:
        text = clean_paragraph(block)
        if len(text) < 90 or not re.search(r"[A-Za-z]{3}", text):
            continue
        if text.count("<!--") or text.startswith(("Operation ", "Figure ")):
            continue
        if len(text) > 1_300:
            # Keep complete sentences only, within the tokenizer-friendly budget.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            selected = []
            for sentence in sentences:
                if selected and sum(map(len, selected)) + len(sentence) > 1_200:
                    break
                selected.append(sentence)
            text = " ".join(selected).strip()
        if len(text) >= 90:
            return text
    return ""


def make_question(title: str) -> str:
    low = title.casefold()
    if low.startswith(("support of ", "support for ")):
        subject = re.sub(r"^support (?:of|for) ", "", title, flags=re.IGNORECASE)
        return f"How does the system support {subject}?"
    if low.startswith(("requirements for ", "requirements on ", "requirements related to ")):
        subject = re.sub(r"^requirements (?:for|on|related to) ", "", title, flags=re.IGNORECASE)
        return f"What requirements apply to {subject}?"
    if "procedure" in low or "procedures" in low:
        return f"What steps and conditions are defined for {title}?"
    if "selection" in low:
        return f"How does the network select or determine {title}?"
    if "registration" in low:
        return f"How does the network handle {title}?"
    if "identification" in low or "identifier" in low or "identity" in low:
        return f"How is {title} identified, represented, or used?"
    if "mobility" in low or "handover" in low:
        return f"How does the system manage {title} during mobility?"
    if "qos" in low or "quality of service" in low:
        return f"What QoS behavior or requirements apply to {title}?"
    if "management" in low or "handling" in low:
        return f"How does the system manage or handle {title}?"
    return f"What does the standard specify about {title}?"


def read_candidates() -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(SOURCE_DIR.glob("*.md")):
        if not path.name.startswith(("ts_", "tr_")):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            match = HEADING_RE.match(line)
            if not match:
                continue
            title = clean_title(match.group("title"))
            clause = (match.group("clause") or "").strip()
            if len(title) < 12 or GENERIC_RE.match(title) or re.match(r"^(?:figure|table)\s", title, re.I):
                continue
            section_lines = []
            for following in lines[i + 1 :]:
                if following.startswith("#"):
                    break
                section_lines.append(following)
            answer = extract_answer(section_lines)
            if not answer:
                continue
            question = make_question(title)
            by_file[path.name].append({
                "question": question,
                "answer": answer,
                "evidence": answer,
                "source_id": f"{path.stem} {clause} {title}".strip(),
            })
    return by_file


def choose_balanced(by_file: dict[str, list[dict]]) -> list[dict]:
    files = sorted(by_file)
    per_file_target = (TOTAL + len(files) - 1) // len(files)
    queues: list[list[dict]] = []
    for name in files:
        candidates = by_file[name]
        # Spread selected clauses across the whole document, rather than using
        # only the first chapters. Take a few more than the equal share for refill.
        count = min(len(candidates), per_file_target + 5)
        selected = []
        for i in range(count):
            idx = min(len(candidates) - 1, int((i + 0.5) * len(candidates) / count))
            selected.append(candidates[idx])
        queues.append(selected)

    chosen: list[dict] = []
    seen: set[str] = set()
    depth = 0
    while len(chosen) < TOTAL:
        advanced = False
        for queue in queues:
            if depth >= len(queue):
                continue
            advanced = True
            row = queue[depth]
            key = re.sub(r"\W+", " ", row["question"].casefold()).strip()
            if key not in seen:
                seen.add(key)
                chosen.append(row)
                if len(chosen) == TOTAL:
                    break
        if not advanced:
            break
        depth += 1

    if len(chosen) < TOTAL:
        for name in files:
            for row in by_file[name]:
                key = re.sub(r"\W+", " ", row["question"].casefold()).strip()
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(row)
                if len(chosen) == TOTAL:
                    break
            if len(chosen) == TOTAL:
                break
    if len(chosen) != TOTAL:
        raise RuntimeError(f"Need {TOTAL} unique QA items; found {len(chosen)}")
    random.Random(SEED).shuffle(chosen)
    return chosen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")
    by_file = read_candidates()
    rows = choose_balanced(by_file)
    train, test = rows[:TRAIN_COUNT], rows[TRAIN_COUNT:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for i, row in enumerate(train, 1):
        train_rows.append({
            "id": f"3gpp-qa-train-{i:04d}",
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        })
    test_rows = []
    eval_questions = []
    gold_rows = []
    for i, row in enumerate(test, 1):
        item_id = f"3gpp-qa-test-{i:04d}"
        test_rows.append({
            "id": item_id,
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        })
        eval_questions.append({"id": item_id, "question": row["question"]})
        gold_rows.append({
            "id": item_id,
            "reference_answer": row["answer"],
            "reference_context": row["evidence"],
            "source_ids": [row["source_id"]],
        })

    write_jsonl(DATA_DIR / "qa_train.jsonl", train_rows)
    write_jsonl(DATA_DIR / "qa_test.jsonl", test_rows)
    write_jsonl(DATA_DIR / "qa_eval.jsonl", eval_questions)
    write_jsonl(DATA_DIR / "qa_eval_gold.jsonl", gold_rows)
    manifest = {
        "total_examples": TOTAL,
        "train_examples": len(train),
        "test_examples": len(test),
        "split": "80/20 by QA item after balanced sampling across source documents",
        "seed": SEED,
        "source_documents": len(by_file),
        "generation_method": "Questions are created from clause headings; reference answers/evidence are the opening substantive passage of that clause.",
        "review_note": "Machine-assembled candidates; expert review is required before treating correctness, groundedness, or hallucination rates as final results.",
        "files": {
            "train": "data/qa_train.jsonl",
            "test_chat": "data/qa_test.jsonl",
            "test_prompts": "data/qa_eval.jsonl",
            "test_gold": "data/qa_eval_gold.jsonl",
        },
    }
    (DATA_DIR / "qa_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(train)} train and {len(test)} held-out QA pairs from {len(by_file)} source documents.")
    print("Train file: data/qa_train.jsonl")
    print("Test prompts: data/qa_eval.jsonl")
    print("Answer/evidence key: data/qa_eval_gold.jsonl")
    for row in eval_questions[:3]:
        print(f"Sample {row['id']}: {row['question']}")


if __name__ == "__main__":
    main()
