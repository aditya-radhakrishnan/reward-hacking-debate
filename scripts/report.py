"""Turn training runs into something readable: a summary table, curves, and transcripts. Free.

Usage:
  python -m scripts.report curves runs/aa-seed0 runs/abab-seed0 --out reports/replication
  python -m scripts.report transcripts runs/abab-seed0 --step 200 -n 5 --out reports/abab-step200.md

`curves` writes <out>/report.md and PNG plots of validation accuracy, judge MCC, the judge's vote
share for Alice, and training reward. `transcripts` writes evaluation episodes as Markdown,
picking judge mistakes first, since those are where reward hacking shows up.
"""

import argparse
import json
from pathlib import Path

from rhd import data

PLOTS = [
    ("eval", "accuracy", "Validation accuracy (Alice's answer vs. reference)"),
    ("eval", "mcc", "Judge MCC on validation (falls when the policy hacks the judge)"),
    ("eval", "alice_share", "Judge vote share for Alice on validation"),
    ("metrics", "reward/alice", "Training reward, Alice"),
    ("metrics", "reward/bob", "Training reward, Bob"),
    ("eval", "bare_answer_rate", "Bare-answer rate on validation"),
]


def load_run(path: Path) -> dict:
    return {
        "name": path.name,
        "config": json.loads((path / "config.json").read_text()),
        "metrics": data.read_jsonl(path / "metrics.jsonl"),
        "eval": data.read_jsonl(path / "eval.jsonl"),
        "usage": json.loads((path / "usage.json").read_text()) if (path / "usage.json").exists() else {},
    }


def _fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def summary_table(runs: list[dict]) -> str:
    lines = ["| Run | Arm | Steps | Spend | Acc. step 0 | Peak acc. (step) | Final acc. | MCC step 0 | Final MCC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        ev = r["eval"]
        if not ev:
            continue
        peak = max(ev, key=lambda e: e["accuracy"] or 0)
        lines.append(
            f"| {r['name']} | {r['config']['arm']} | {r['metrics'][-1]['step'] if r['metrics'] else 0} | "
            f"${r['usage'].get('cost_usd', 0):,.0f} | {_fmt(ev[0]['accuracy'])} | "
            f"{_fmt(peak['accuracy'])} ({peak['step']}) | {_fmt(ev[-1]['accuracy'])} | "
            f"{_fmt(ev[0]['mcc'])} | {_fmt(ev[-1]['mcc'])} |")
    return "\n".join(lines)


def curves(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_run(Path(p)) for p in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Training report", "", summary_table(runs), ""]
    for source, key, title in PLOTS:
        series = [(r["name"], [(row["step"], row[key]) for row in r[source] if row.get(key) is not None])
                  for r in runs]
        series = [(n, pts) for n, pts in series if pts]
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, pts in series:
            ax.plot(*zip(*pts), marker="o" if source == "eval" else None, markersize=3, label=name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = key.replace("/", "_") + ".png"
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
        md += [f"## {title}", "", f"![{title}]({fname})", ""]
    (out / "report.md").write_text("\n".join(md))
    print(f"wrote {out / 'report.md'}")


def episode_markdown(e: dict, title: str) -> str:
    votes = e["verdicts"]
    lines = [f"### {title}", "",
             f"**Reference answer:** `{e['answer']}` · **Alice correct:** {e['alice_correct']} · "
             f"**Judge votes for CORRECT:** {sum(v is True for v in votes)}/{len(votes)} · "
             f"**Rewards by ply:** {', '.join(f'{r:.2f}' for r in e['rewards'])}", "",
             "**Problem**", "", e["problem"], ""]
    for t in e["turns"]:
        who = {"solve": "Alice (solution)", "critique": "Bob (critique)", "rebuttal": "Alice (reply)"}[t["kind"]]
        lines += [f"**{who}** · {t['n_tokens']} tokens, {t['reasoning_chars']:,} chars of hidden reasoning"
                  f"{'' if t['ok'] else ' · FORMAT FAILURE'}", "", t["visible"] or "_(no visible response)_", ""]
    for i, text in enumerate(e.get("judge_samples", []), 1):
        quoted = "\n".join("> " + line for line in (text or "(no visible response)").splitlines())
        lines += [f"**Judge, vote {i}**", "", quoted, ""]
    return "\n".join(lines)


def transcripts(args) -> None:
    run = Path(args.runs[0])
    steps = sorted(int(f.stem[4:]) for f in (run / "transcripts/eval").glob("step*.jsonl"))
    if not steps:
        raise SystemExit(f"no evaluation transcripts in {run}")
    step = args.step if args.step is not None else steps[-1]
    eps = data.read_jsonl(run / f"transcripts/eval/step{step:04d}.jsonl")
    judged = [e for e in eps if e["verdicts"]]

    def fooled(e):  # the judge's majority disagrees with the reference
        return (e["alice_share"] >= 0.5) != e["alice_correct"]

    picked = sorted(judged, key=lambda e: (not fooled(e), not (e["alice_share"] >= 0.5)))[: args.n]
    md = [f"# {run.name}: evaluation transcripts at step {step}", "",
          f"{len(picked)} of {len(eps)} episodes, judge mistakes first "
          f"({sum(map(fooled, judged))} of {len(judged)} judged episodes were mistakes).", ""]
    for i, e in enumerate(picked, 1):
        kind = ("Judge fooled: accepted a wrong answer" if e["alice_share"] >= 0.5 else
                "Judge fooled: rejected a right answer") if fooled(e) else "Judge right"
        md += [episode_markdown(e, f"{i}. {kind}"), "---", ""]
    out = Path(args.out or run / f"transcripts_step{step:04d}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["curves", "transcripts"])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=None, help="transcripts: evaluation step (default: last)")
    ap.add_argument("-n", type=int, default=5, help="transcripts: how many episodes")
    args = ap.parse_args()
    if args.what == "curves":
        args.out = args.out or "reports/latest"
        curves(args)
    else:
        transcripts(args)


if __name__ == "__main__":
    main()
