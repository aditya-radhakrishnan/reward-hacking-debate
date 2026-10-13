"""Train one arm (rlvr, aa, abab, abaaba). Without --yes this only prints the cost estimate.

Usage:
  python -m scripts.train abab                                  # free: print the cost estimate
  python -m scripts.train abab --yes --max-usd 2100             # paid run, stops at the cap
  python -m scripts.train rlvr --steps 30 --lr 2e-5 --run-dir runs/lr-2e-5 --yes --max-usd 100

A run resumes from its last checkpoint if its run directory already exists. See docs/DESIGN.md.
"""

import argparse
import asyncio
import sys
from dataclasses import fields

from rhd.costs import estimate
from rhd.debate import PROTOCOLS
from rhd.trainer import RunConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=list(PROTOCOLS))
    ap.add_argument("--yes", action="store_true", help="confirm spending Tinker credits")
    ap.add_argument("--run-dir", default=None, help="default runs/<arm>-seed<seed>")
    for f in fields(RunConfig):
        if f.name in ("arm", "run_dir"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = ap.parse_args()

    run_dir = args.run_dir or f"runs/{args.arm}-seed{args.seed}"
    kw = {f.name: getattr(args, f.name) for f in fields(RunConfig) if f.name not in ("arm", "run_dir")}
    cfg = RunConfig(arm=args.arm, run_dir=run_dir, **kw)

    est = estimate(cfg)
    print(f"{cfg.arm}: {cfg.steps} steps x {cfg.groups_per_step} groups x {cfg.group_size}, "
          f"{cfg.votes} votes, eval every {cfg.eval_every} on {cfg.eval_problems}x{cfg.eval_samples}")
    print("\n".join(est.lines()))
    if not args.yes:
        print("\nNo credits spent. Re-run with --yes --max-usd <cap> to train.")
        return
    if cfg.max_usd <= 0:
        sys.exit("a paid run needs --max-usd; the run stops before a step would pass it")
    asyncio.run(train(cfg))


if __name__ == "__main__":
    main()
