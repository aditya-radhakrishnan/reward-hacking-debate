"""Runs the paid pilot stages end to end against a fake model (no API key, no spend)."""

import argparse
import asyncio
import random

import pytest

import scripts.pilot as pilot
from rhd import data
from rhd.sampling import Completion


class FakeHandle:
    """Policy answers correctly half the time; judge says CORRECT iff the answer is right, 80% of
    the time."""

    def __init__(self, spec, usage):
        self.spec, self.usage, self.rng = spec, usage, random.Random(0)

    @classmethod
    async def create(cls, service, spec, usage, concurrency=64):
        return cls(spec, usage)

    async def sample(self, messages, n, force_prefix=None):
        self.usage.sampled[self.spec.base_model] += 100 * n
        user = messages[-1]["content"]
        if user.startswith("## Problem"):
            right = "\\boxed{42}" in user
            verdicts = ["CORRECT" if (right == (self.rng.random() < 0.8)) else "INCORRECT" for _ in range(n)]
            return [Completion("", f"VERDICT: {v}", 100, True) for v in verdicts]
        return [Completion("r", f"\\boxed{{{42 if self.rng.random() < 0.5 else 7}}}", 100, True) for _ in range(n)]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    out = tmp_path / "pilot"
    for name, path in {
        "OUT": out, "POOL": out / "pool.jsonl", "POLICY_SAMPLES": out / "policy_samples.jsonl",
        "FILTERED": out / "filtered.jsonl", "USAGE": out / "usage.json",
    }.items():
        monkeypatch.setattr(pilot, name, path)
    monkeypatch.setattr(pilot, "ModelHandle", FakeHandle)
    monkeypatch.setattr(pilot.tinker, "ServiceClient", lambda: None)
    out.mkdir()
    data.write_jsonl(out / "pool.jsonl", [
        {"id": f"p{i}", "source": "dapo", "problem": f"problem {i}", "answer": "42"} for i in range(40)
    ])
    return out


def args(**kw):
    base = dict(policy="gptoss20b-medium", limit=None, policy_samples=4, min_pass=0.2, max_pass=0.8, concurrency=8,
                judges=["qwen3-8b-nothink"], judge_problems=300, judge_solve_samples=2, votes=8)
    return argparse.Namespace(**{**base, **kw})


def test_filter_judge_report(workdir, capsys):
    asyncio.run(pilot.stage_filter(args()))
    filtered = data.read_jsonl(pilot.FILTERED)
    assert 0 < len(filtered) < 40
    assert all(0.2 <= r["pass_rate"] <= 0.8 for r in filtered)

    asyncio.run(pilot.stage_judge(args()))
    rep = pilot.judge_report("qwen3-8b-nothink", policy_acc=0.5)
    assert rep["solutions_judged"] == 4 * len(filtered)
    assert rep["mcc_majority"] > 0.5 and rep["unparsed_vote_rate"] == 0

    # Re-running resumes: no new rows, spend recorded.
    asyncio.run(pilot.stage_filter(args()))
    assert len(data.read_jsonl(pilot.POLICY_SAMPLES)) == 40
    pilot.stage_report(args())
    out = capsys.readouterr().out
    assert "Judges at step 0" in out and "Total pilot spend" in out
