"""Load and normalize public math datasets into {id, source, problem, answer} records."""

import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

_DAPO_PREFIX = re.compile(r"^Solve the following math problem step by step\..*?\n\n", re.S)
_DAPO_SUFFIX = re.compile(r'\n\nRemember to put your answer on its own line after "Answer:"\.?\s*$')

# Reference answers the grader can't check reliably: prose, yes/no, sets and intervals,
# inequalities, multi-part answers, and plain words (3+ letters not part of a LaTeX command).
_UNCHECKABLE = re.compile(
    r"\\text|\\mathrm|\\cup|\\cap|\\in\b|\\le|\\ge|[<>;]|\),\s*\(|(?<![\\a-zA-Z])[a-zA-Z]{3,}"
)


def is_checkable_reference(answer: str) -> bool:
    return bool(answer.strip()) and not _UNCHECKABLE.search(answer)


def _record(source: str, problem: str, answer: str) -> dict:
    problem, answer = problem.strip(), str(answer).strip()
    uid = hashlib.sha1(problem.encode()).hexdigest()[:12]
    return {"id": uid, "source": source, "problem": problem, "answer": answer}


def strip_dapo_prompt(text: str) -> str:
    return _DAPO_SUFFIX.sub("", _DAPO_PREFIX.sub("", text))


def load_dapo() -> list[dict]:
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    return [
        _record("dapo", strip_dapo_prompt(r["prompt"][0]["content"]), r["reward_model"]["ground_truth"])
        for r in ds
    ]


def load_deepscaler() -> list[dict]:
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    return [_record("deepscaler", r["problem"], r["answer"]) for r in ds]


def load_omni(min_difficulty: float = 6.0) -> list[dict]:
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    return [
        _record("omni", r["problem"], r["answer"])
        for r in ds
        if r["difficulty"] >= min_difficulty
    ]


LOADERS = {"dapo": load_dapo, "deepscaler": load_deepscaler, "omni": load_omni}


def build_pool(n: int, seed: int = 0, sources: list[str] = ("dapo", "deepscaler", "omni")) -> list[dict]:
    """Sample n problems, split evenly across sources, deduplicated by problem text."""
    sources = [LOADERS[name]() for name in sources]
    rng = random.Random(seed)
    seen, pool = set(), []
    per_source = n // len(sources)
    for records in sources:
        rng.shuffle(records)
        taken = 0
        for r in records:
            if taken == per_source:
                break
            if r["id"] in seen or not is_checkable_reference(r["answer"]):
                continue
            seen.add(r["id"])
            pool.append(r)
            taken += 1
    rng.shuffle(pool)
    return pool


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def is_english(problem: str) -> bool:
    """DAPO-Math-17k includes Chinese problems (29% of our filtered set). Whether to keep them is
    still open (docs/DESIGN.md, "Data"); they are left out by default for now."""
    return not _CJK.search(problem)


def train_val_split(rows: list[dict], seed: int = 0, english_only: bool = True) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 IID split by problem id."""
    rows = sorted((r for r in rows if not english_only or is_english(r["problem"])), key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    half = len(rows) // 2
    return rows[:half], rows[half:]
