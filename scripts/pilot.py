"""Gate 0 pilot (plan section 4d): no training, sampling only.

Stages (run in order; each is resumable and writes under runs/pilot/):
  pool    Build a ~3k problem pool from DAPO / DeepScaleR / Omni-MATH.            (free)
  filter  Policy samples each pool problem 4x; keep pass rate in [0.2, 0.8].     (~$30)
  judge   For each judge candidate, on the filtered set: judge solve rate, and    (~$5-15
          8 votes on each policy solution -> MCC / AUROC at step 0.                per judge)
  regrade Re-check saved policy solutions with the current grader.               (free)
  report  Print metrics and the Gate 0 verdict.                                   (free)

Usage:
  python -m scripts.pilot pool
  python -m scripts.pilot filter --yes
  python -m scripts.pilot judge --yes [--judges qwen3-8b-nothink gptoss20b-low]
  python -m scripts.pilot report
"""

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import tinker

from rhd import data
from rhd.config import JUDGE_VOTES, JUDGES, POLICIES, POLICY
from rhd.grading import is_correct
from rhd.judge import judge_messages, parse_verdict, solver_messages
from rhd.metrics import auroc, mcc
from rhd.sampling import ModelHandle, Usage

OUT = Path("runs/pilot")
POOL = OUT / "pool.jsonl"
POLICY_SAMPLES = OUT / "policy_samples.jsonl"
FILTERED = OUT / "filtered.jsonl"
USAGE = OUT / "usage.json"


def set_out_dir(out: Path) -> None:
    global OUT, POOL, POLICY_SAMPLES, FILTERED, USAGE
    OUT = out
    POOL = OUT / "pool.jsonl"
    POLICY_SAMPLES = OUT / "policy_samples.jsonl"
    FILTERED = OUT / "filtered.jsonl"
    USAGE = OUT / "usage.json"


def judge_dir(judge: str) -> Path:
    return OUT / "judges" / judge


class Appender:
    """Append-only JSONL writer so an interrupted stage resumes where it stopped."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {r["key"] for r in data.read_jsonl(path)}
        self.f = path.open("a")

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()


def record_usage(usage: Usage, stage: str) -> None:
    all_usage = json.loads(USAGE.read_text()) if USAGE.exists() else {}
    prev = all_usage.get(stage, {"cost_usd": 0.0})
    now = usage.summary()
    now["cost_usd"] = round(now["cost_usd"] + prev["cost_usd"], 2)
    all_usage[stage] = now
    USAGE.write_text(json.dumps(all_usage, indent=2))
    print(f"[{stage}] this run ${usage.cost_usd():.2f}; stage total ${now['cost_usd']:.2f}")


async def gather_with_progress(coros, label: str) -> None:
    total, done = len(coros), 0
    for fut in asyncio.as_completed(coros):
        await fut
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {label}: {done}/{total}", flush=True)


def stage_pool(args) -> None:
    pool = data.build_pool(args.pool_size, seed=args.seed, sources=args.sources)
    data.write_jsonl(POOL, pool)
    by_source = {s: sum(r["source"] == s for r in pool) for s in {r["source"] for r in pool}}
    print(f"wrote {len(pool)} problems to {POOL}: {by_source}")


async def stage_filter(args) -> None:
    pool = data.read_jsonl(POOL)[: args.limit]
    out = Appender(POLICY_SAMPLES)
    todo = [p for p in pool if p["id"] not in out.done]
    print(f"{len(todo)} problems x {args.policy_samples} samples "
          f"({len(pool) - len(todo)} already done)")
    usage = Usage()
    spec = POLICIES[args.policy]
    print(f"policy {spec.name}")
    policy = await ModelHandle.create(tinker.ServiceClient(), spec, usage, args.concurrency)

    async def one(p):
        comps = await policy.sample(solver_messages(p["problem"]), args.policy_samples)
        out.write({
            "key": p["id"],
            **p,
            "solutions": [
                {"visible": c.visible, "n_tokens": c.n_tokens, "clean": c.clean,
                 "correct": is_correct(c.visible, p["answer"])}
                for c in comps
            ],
        })

    try:
        await gather_with_progress([one(p) for p in todo], "filter")
    finally:
        record_usage(usage, "filter")
    write_filtered(args)


def write_filtered(args) -> None:
    rows = data.read_jsonl(POLICY_SAMPLES)
    keep = []
    for r in rows:
        r["pass_rate"] = statistics.mean(s["correct"] for s in r["solutions"])
        if data.is_checkable_reference(r["answer"]) and args.min_pass <= r["pass_rate"] <= args.max_pass:
            keep.append(r)
    data.write_jsonl(FILTERED, keep)
    print(f"kept {len(keep)}/{len(rows)} problems with pass rate in [{args.min_pass}, {args.max_pass}] "
          f"-> {FILTERED}")


async def stage_judge(args) -> None:
    problems = data.read_jsonl(FILTERED)[: args.judge_problems]
    if not problems:
        sys.exit(f"{FILTERED} is empty; run the filter stage first")
    service = tinker.ServiceClient()
    for judge_name in args.judges:
        spec = JUDGES[judge_name]
        usage = Usage()
        judge = await ModelHandle.create(service, spec, usage, args.concurrency)
        solves = Appender(judge_dir(judge_name) / "solves.jsonl")
        votes = Appender(judge_dir(judge_name) / "votes.jsonl")

        async def solve(p):
            comps = await judge.sample(solver_messages(p["problem"]), args.judge_solve_samples)
            solves.write({"key": p["id"], "correct": [is_correct(c.visible, p["answer"]) for c in comps]})

        async def vote(p, i, sol):
            comps = await judge.sample(judge_messages(p["problem"], sol["visible"]), args.votes)
            verdicts = [parse_verdict(c.visible) for c in comps]
            votes.write({
                "key": f"{p['id']}:{i}", "problem_id": p["id"], "correct": sol["correct"],
                "verdicts": verdicts, "judge_tokens": [c.n_tokens for c in comps],
            })

        coros = [solve(p) for p in problems if p["id"] not in solves.done]
        for p in problems:
            for i, sol in enumerate(p["solutions"]):
                # An empty visible response (policy truncated mid-reasoning) is trivially wrong and
                # would inflate MCC, so leave it out.
                if sol["visible"] and f"{p['id']}:{i}" not in votes.done:
                    coros.append(vote(p, i, sol))
        print(f"judge {judge_name}: {len(coros)} sampling calls")
        try:
            await gather_with_progress(coros, judge_name)
        finally:
            record_usage(usage, f"judge:{judge_name}")


def stage_regrade(args) -> None:
    """Re-check saved policy solutions with the current grader (free) and rebuild the filtered set."""
    rows = data.read_jsonl(POLICY_SAMPLES)
    changed = 0
    for r in rows:
        for sol in r["solutions"]:
            if not sol["clean"]:
                sol["visible"] = ""  # saved before truncated reasoning was blanked in sampling.py
            now = is_correct(sol["visible"], r["answer"])
            changed += now != sol["correct"]
            sol["correct"] = now
    data.write_jsonl(POLICY_SAMPLES, rows)
    print(f"regraded {sum(len(r['solutions']) for r in rows)} solutions; {changed} changed")
    write_filtered(args)


def judge_report(judge_name: str, policy_acc: float) -> dict | None:
    votes = data.read_jsonl(judge_dir(judge_name) / "votes.jsonl")
    solves = data.read_jsonl(judge_dir(judge_name) / "solves.jsonl")
    if not votes:
        return None
    labels = [v["correct"] for v in votes]
    shares, per_vote_preds, per_vote_labels, unparsed = [], [], [], 0
    for v in votes:
        parsed = [x for x in v["verdicts"] if x is not None]
        unparsed += len(v["verdicts"]) - len(parsed)
        shares.append(statistics.mean(parsed) if parsed else 0.0)
        per_vote_preds += parsed
        per_vote_labels += [v["correct"]] * len(parsed)
    n_votes = sum(len(v["verdicts"]) for v in votes)
    return {
        "judge": judge_name,
        "solutions_judged": len(votes),
        "frac_correct": round(statistics.mean(labels), 3),
        "mcc_majority": round(mcc([s >= 0.5 for s in shares], labels), 3),
        "mcc_per_vote": round(mcc(per_vote_preds, per_vote_labels), 3),
        "auroc_vote_share": round(auroc(shares, labels), 3),
        "unparsed_vote_rate": round(unparsed / n_votes, 3),
        "judge_solve_rate": round(statistics.mean(c for s in solves for c in s["correct"]), 3) if solves else None,
        "policy_solve_rate": round(policy_acc, 3),
        "mean_judge_tokens": round(statistics.mean(t for v in votes for t in v["judge_tokens"])),
    }


def stage_report(args) -> None:
    rows = data.read_jsonl(POLICY_SAMPLES)
    filtered = data.read_jsonl(FILTERED)
    if not rows:
        sys.exit("no policy samples yet; run the filter stage first")
    sols = [s for r in rows for s in r["solutions"]]
    print("== Policy on the full pool")
    print(f"  problems {len(rows)}, solve rate {statistics.mean(s['correct'] for s in sols):.3f}, "
          f"truncated/malformed {statistics.mean(not s['clean'] for s in sols):.3f}, "
          f"mean tokens {statistics.mean(s['n_tokens'] for s in sols):.0f}")
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(statistics.mean(s["correct"] for s in r["solutions"]))
    for src, rates in sorted(by_source.items()):
        print(f"  {src:<11} solve rate {statistics.mean(rates):.3f}  kept {sum(0.2 <= x <= 0.8 for x in rates)}/{len(rates)}")
    print(f"  filtered set: {len(filtered)} problems (target 1-2k)")

    judged_ids = {p["id"] for p in filtered[: args.judge_problems]}
    policy_acc = statistics.mean(
        s["correct"] for p in filtered if p["id"] in judged_ids for s in p["solutions"]
    ) if judged_ids else float("nan")

    print("\n== Judges at step 0 (Gate 0: MCC 0.4-0.7, judge solve rate well below policy)")
    for name in JUDGES:
        rep = judge_report(name, policy_acc)
        if rep is None:
            continue
        ok = 0.4 <= rep["mcc_majority"] <= 0.7 and (rep["judge_solve_rate"] or 1) < policy_acc - 0.15
        print(json.dumps(rep))
        print(f"  -> {'PASS' if ok else 'CHECK'}")
    if USAGE.exists():
        total = sum(v["cost_usd"] for v in json.loads(USAGE.read_text()).values())
        print(f"\nTotal pilot spend so far: ${total:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pool", "filter", "judge", "regrade", "report"])
    ap.add_argument("--yes", action="store_true", help="confirm a stage that spends Tinker credits")
    ap.add_argument("--out", default=None, help="output directory (default runs/pilot)")
    ap.add_argument("--policy", default=POLICY.name, choices=list(POLICIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=3000)
    ap.add_argument("--sources", nargs="+", default=list(data.LOADERS), choices=list(data.LOADERS))
    ap.add_argument("--limit", type=int, default=None, help="only process the first N pool problems")
    ap.add_argument("--policy-samples", type=int, default=4)
    ap.add_argument("--min-pass", type=float, default=0.2)
    ap.add_argument("--max-pass", type=float, default=0.8)
    ap.add_argument("--judges", nargs="+", default=["qwen3-8b-nothink", "gptoss20b-low"], choices=list(JUDGES))
    ap.add_argument("--judge-problems", type=int, default=300)
    ap.add_argument("--judge-solve-samples", type=int, default=2)
    ap.add_argument("--votes", type=int, default=JUDGE_VOTES, help="judge votes per solution")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    if args.stage in ("filter", "judge") and not args.yes:
        sys.exit(f"'{args.stage}' spends Tinker credits (see plan section 4d); re-run with --yes")
    if args.out:
        set_out_dir(Path(args.out))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "pool":
        stage_pool(args)
    elif args.stage == "filter":
        asyncio.run(stage_filter(args))
    elif args.stage == "judge":
        asyncio.run(stage_judge(args))
    elif args.stage == "regrade":
        stage_regrade(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
