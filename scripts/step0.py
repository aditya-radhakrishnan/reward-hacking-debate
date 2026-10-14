"""Summarise the judge's step-0 accuracy on the committed dataset, as the training loop measures it.

Reads the policy solutions in runs/data_dapo/filtered.jsonl and the judge votes written by
`python -m scripts.pilot judge --out runs/data_dapo --judges <judge>`, and writes
runs/data_dapo/judges/<judge>/summary.json. Free.

Matches rhd/trainer.py: a vote with no verdict counts as INCORRECT, the judge's prediction is its
majority (vote share >= 0.5), and solutions that fail the format rule are left out. Confidence
intervals resample problems rather than solutions, since a problem's solutions are correlated.

Usage:
  python -m scripts.step0 [--judge gptoss20b-medium-24k]
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from rhd import data
from rhd.config import JUDGE
from rhd.debate import format_ok, is_bare_answer
from rhd.metrics import auroc, mcc

DATA = Path("runs/data_dapo")


def _metrics(rows: list[dict]) -> dict:
    labels = [r["correct"] for r in rows]
    shares = [r["share"] for r in rows]
    return {"mcc": mcc([s >= 0.5 for s in shares], labels), "auroc": auroc(shares, labels)}


def _ci(by_problem: dict[str, list[dict]], key: str, n: int = 2000, seed: int = 0) -> list[float]:
    rng, ids, vals = random.Random(seed), list(by_problem), []
    for _ in range(n):
        sample = [r for pid in rng.choices(ids, k=len(ids)) for r in by_problem[pid]]
        val = _metrics(sample)[key]
        if not math.isnan(val):  # AUROC is undefined when a resample has only one class
            vals.append(val)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]


def summarise(rows: list[dict]) -> dict:
    by_problem: dict[str, list[dict]] = {}
    for r in rows:
        by_problem.setdefault(r["problem_id"], []).append(r)
    m = _metrics(rows)
    votes = [v for r in rows for v in r["verdicts"]]
    return {
        "problems": len(by_problem),
        "solutions": len(rows),
        "policy_accuracy": round(float(statistics.mean(r["correct"] for r in rows)), 3),
        "mcc": round(m["mcc"], 3),
        "mcc_95ci": _ci(by_problem, "mcc"),
        "auroc": round(m["auroc"], 3),
        "auroc_95ci": _ci(by_problem, "auroc"),
        "judge_says_correct": round(float(statistics.mean(r["share"] >= 0.5 for r in rows)), 3),
        "no_verdict_rate": round(float(statistics.mean(v is None for v in votes)), 4),
        "forced_vote_rate": round(float(statistics.mean(f for r in rows for f in r["forced"])), 4),
        "bare_answer_rate": round(float(statistics.mean(r["bare"] for r in rows)), 3),
        "mean_judge_tokens": round(statistics.mean(t for r in rows for t in r["judge_tokens"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default=JUDGE.name)
    args = ap.parse_args()

    out_dir = DATA / "judges" / args.judge
    problems = {p["id"]: p for p in data.read_jsonl(DATA / "filtered.jsonl")}
    votes = data.read_jsonl(out_dir / "votes.jsonl")
    solves = data.read_jsonl(out_dir / "solves.jsonl")
    if not votes:
        raise SystemExit(f"no votes in {out_dir}; run the pilot judge stage first")

    rows, excluded = [], 0
    for v in votes:
        p = problems[v["problem_id"]]
        sol = p["solutions"][int(v["key"].rsplit(":", 1)[1])]
        if not format_ok("solve", sol["clean"], sol["visible"]):
            excluded += 1
            continue
        rows.append({
            "problem_id": p["id"], "correct": sol["correct"], "verdicts": v["verdicts"],
            "share": sum(x is True for x in v["verdicts"]) / len(v["verdicts"]),
            "judge_tokens": v["judge_tokens"], "bare": is_bare_answer(sol["visible"]),
            "forced": v.get("forced", [False] * len(v["verdicts"])),
            "english": data.is_english(p["problem"]),
        })

    summary = {
        "judge": args.judge,
        "all": summarise(rows),
        "english": summarise([r for r in rows if r["english"]]),
        "chinese": summarise([r for r in rows if not r["english"]]),
        "excluded_format_failures": excluded,
        "judge_solve_rate": round(float(statistics.mean(c for s in solves for c in s["correct"])), 3) if solves else None,
        "policy_solve_rate_same_problems": round(float(statistics.mean(
            s["correct"] for pid in {s["key"] for s in solves} for s in problems[pid]["solutions"])), 3) if solves else None,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
