"""The training loop end to end against a fake Tinker backend (no API key, no spend), plus unit
tests for the protocols and rewards."""

import asyncio
import json
import random

import pytest
from tinker import types

import rhd.trainer as trainer
from rhd import data, debate
from rhd.costs import estimate
from rhd.judge import JUDGE_SYSTEM
from rhd.rollout import Settings, play_group
from rhd.sampling import Completion


# --- fakes -------------------------------------------------------------------------------------

class FakeHandle:
    """Alice is right half the time; Bob always objects; the judge agrees with the truth 80% of
    the time. Every completion carries tokens and logprobs so datums can be built."""

    calls: list[tuple[str, int]] = []

    def __init__(self, spec, client, usage, concurrency=64):
        self.spec, self.client, self.usage, self.rng = spec, client, usage, random.Random(0)

    @classmethod
    async def create(cls, service, spec, usage, concurrency=64):
        return cls(spec, None, usage, concurrency)

    async def sample(self, messages, n):
        system, user = messages[0]["content"], messages[-1]["content"]
        self.usage.sampled[self.spec.base_model] += 100 * n
        if system in (JUDGE_SYSTEM, debate.DEBATE_JUDGE_SYSTEM):
            kind = "judge"
            solution = user.split("## Solution\n" if system == JUDGE_SYSTEM else "## Alice's solution\n")[1]
            right = "\\boxed{42}" in solution.split("## ")[0]
            texts = [f"VERDICT: {'CORRECT' if right == (self.rng.random() < 0.8) else 'INCORRECT'}" for _ in range(n)]
        elif system.startswith("You are Bob"):
            kind = "critique"
            texts = ["Step two divides by zero. " * self.rng.choice([1, 40]) for _ in range(n)]
        elif system.startswith("You are Alice"):
            kind = "rebuttal"
            texts = ["Step two is fine." for _ in range(n)]
        else:
            kind = "solve"
            texts = [f"Factor and compare terms. \\boxed{{{self.rng.choice([42, 7])}}}" for _ in range(n)]
        FakeHandle.calls.append((kind, n))
        prompt = types.ModelInput.from_ints(list(range(10)))
        out = []
        for text in texts:
            clean = kind == "judge" or self.rng.random() > 0.05
            out.append(Completion("hidden reasoning", text if clean else "", 50, clean, prompt=prompt,
                                  tokens=[1, 2, 3, 4, 5], logprobs=[-0.5] * 5))
        return out


class _Done:
    def __init__(self, value=None):
        self.value = value

    async def result_async(self):
        return self.value


class FakeTrainingClient:
    def __init__(self, path=None):
        self.loaded_from, self.optim_steps, self.datums = path, 0, []

    async def forward_backward_async(self, datums, loss_fn):
        assert loss_fn == "importance_sampling"
        for d in datums:
            n = d.model_input.length
            assert all(len(t.tolist()) == n for t in d.loss_fn_inputs.values())
        self.datums.append(datums)
        return _Done()

    async def optim_step_async(self, adam):
        self.optim_steps += 1
        self.lr = adam.learning_rate
        return _Done()

    async def save_weights_and_get_sampling_client_async(self):
        return object()

    async def save_state_async(self, name):
        return _Done(types.SaveWeightsResponse(path=f"tinker://fake/{name}"))


class FakeService:
    def __init__(self):
        self.clients = []

    async def create_lora_training_client_async(self, base_model, rank):
        self.clients.append(FakeTrainingClient())
        return self.clients[-1]

    async def create_training_client_from_state_with_optimizer_async(self, path):
        self.clients.append(FakeTrainingClient(path))
        return self.clients[-1]


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "ModelHandle", FakeHandle)
    FakeHandle.calls = []
    rows = [{"id": f"p{i:02d}", "source": "dapo", "problem": f"Problem {i}?", "answer": "42",
             "pass_rate": 0.5} for i in range(40)]
    data.write_jsonl(tmp_path / "filtered.jsonl", rows)
    return tmp_path


def cfg(tmp_path, arm, **kw):
    base = dict(arm=arm, run_dir=str(tmp_path / arm), steps=3, groups_per_step=4, group_size=4,
                eval_every=2, save_every=1, eval_problems=5, eval_samples=1, max_usd=1000.0,
                data_path=str(tmp_path / "filtered.jsonl"), concurrency=4)
    return trainer.RunConfig(**{**base, **kw})


# --- end to end --------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(debate.PROTOCOLS))
def test_every_arm_trains_and_logs(fake, arm):
    service = FakeService()
    c = cfg(fake, arm)
    asyncio.run(trainer.train(c, service))
    run = fake / arm

    metrics = data.read_jsonl(run / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [1, 2, 3]
    assert [e["step"] for e in data.read_jsonl(run / "eval.jsonl")] == [0, 2, 3]
    assert (run / "transcripts/eval/step0000.jsonl").exists()
    assert (run / "transcripts/train/step0001.jsonl").exists()
    assert json.loads((run / "usage.json").read_text())["cost_usd"] > 0

    tc = service.clients[0]
    assert tc.optim_steps == 3
    assert tc.lr == pytest.approx(c.lr * debate.PROTOCOLS[arm].lr_scale)
    if arm in ("abab", "abaaba"):
        assert any(m["reward/bob"] is not None for m in metrics)
    if arm == "rlvr":
        # Only the 3 evaluations (5 episodes each, fewer if a solution fails the format rule) call
        # the judge; training never does.
        assert 0 < sum(k == "judge" for k, _ in FakeHandle.calls) <= 3 * 5
        assert data.read_jsonl(run / "eval.jsonl")[0]["judged"] > 0


def test_resume_continues_from_checkpoint(fake):
    service = FakeService()
    asyncio.run(trainer.train(cfg(fake, "abab", steps=2), service))
    asyncio.run(trainer.train(cfg(fake, "abab", steps=4), service))
    run = fake / "abab"
    assert service.clients[1].loaded_from == "tinker://fake/abab-step0002"
    assert [m["step"] for m in data.read_jsonl(run / "metrics.jsonl")] == [1, 2, 3, 4]
    assert [e["step"] for e in data.read_jsonl(run / "eval.jsonl")] == [0, 2, 4]


def test_resume_refuses_changed_settings(fake):
    asyncio.run(trainer.train(cfg(fake, "aa", steps=1), FakeService()))
    with pytest.raises(SystemExit, match="votes"):
        asyncio.run(trainer.train(cfg(fake, "aa", steps=2, votes=4), FakeService()))


def test_spend_cap_stops_before_spending(fake):
    service = FakeService()
    asyncio.run(trainer.train(cfg(fake, "aa", max_usd=1e-6), service))
    assert not (fake / "aa" / "metrics.jsonl").exists()
    assert service.clients[0].optim_steps == 0


# --- pieces ------------------------------------------------------------------------------------

def test_prefix_then_diverge_shares_the_prefix(fake):
    FakeHandle.calls = []
    from rhd.sampling import Usage
    from rhd.config import POLICY, JUDGE
    u = Usage()
    policy, judge = FakeHandle(POLICY, None, u), FakeHandle(JUDGE, None, u)
    problem = {"id": "x", "problem": "P?", "answer": "42"}
    random.seed(1)
    for _ in range(20):  # retry until the one-shot prefix passes the format rule
        FakeHandle.calls = []
        group = asyncio.run(play_group(policy, judge, debate.PROTOCOLS["abaaba"], problem, 1, 4, Settings(votes=3)))
        if group:
            break
    assert FakeHandle.calls[:2] == [("solve", 1), ("critique", 4)]
    assert all(e.turns[0] is group[0].turns[0] for e in group)
    assert len({id(e.turns[1]) for e in group}) == 4


def test_judge_never_sees_hidden_reasoning():
    msgs = debate.debate_judge_messages("P?", ["Alice says \\boxed{1}", "Bob objects", "Alice replies"])
    text = json.dumps(msgs)
    assert "Alice says" in text and "Bob objects" in text and "Alice replies" in text
    assert "hidden" not in text


def test_rewards():
    kw = dict(word_limit=10, per_word=0.01)
    assert debate.ply_reward("solve", True, "x \\boxed{1}", 0.75, **kw) == 0.75
    assert debate.ply_reward("critique", True, "short", 0.75, **kw) == 0.25
    assert debate.ply_reward("critique", True, "w " * 30, 0.75, **kw) == pytest.approx(0.25 - 0.20)
    assert debate.ply_reward("rebuttal", True, "w " * 500, 1.0, **kw) == 0.0  # penalty capped at 1
    assert debate.ply_reward("critique", False, "", 0.0, **kw) == -1.0
    assert not debate.format_ok("solve", True, "no boxed answer")
    assert debate.format_ok("critique", True, "no boxed answer needed")
    assert not debate.format_ok("solve", False, "\\boxed{1}")


def test_bare_answer():
    assert debate.is_bare_answer("\\boxed{11}")
    assert debate.is_bare_answer("So k = \\boxed{11}")
    assert not debate.is_bare_answer("Let x be the side. Then the volume is sqrt(11), so \\boxed{11}")


def test_batches_are_deterministic_and_cover_epochs():
    train = [{"id": i} for i in range(10)]
    first = [p["id"] for s in range(5) for p in trainer.batch_problems(train, s, 4, seed=0)]
    assert sorted(first[:10]) == list(range(10)) and sorted(first[10:20]) == list(range(10))
    assert first == [p["id"] for s in range(5) for p in trainer.batch_problems(train, s, 4, seed=0)]


def test_split_is_english_and_disjoint():
    rows = [{"id": f"e{i}", "problem": "Find x."} for i in range(10)] + [{"id": "c", "problem": "求x的值"}]
    train, val = data.train_val_split(rows)
    assert len(train) == len(val) == 5
    assert not {r["id"] for r in train} & {r["id"] for r in val}


def test_make_datum_masks_the_prompt():
    c = Completion("", "v", 3, True, prompt=types.ModelInput.from_ints([9, 9, 9, 9]),
                   tokens=[1, 2, 3], logprobs=[-1.0, -2.0, -3.0])
    d = trainer.make_datum(c, 0.5)
    assert d.model_input.to_ints() == [9, 9, 9, 9, 1, 2]
    assert d.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 0, 1, 2, 3]
    assert d.loss_fn_inputs["advantages"].tolist() == [0, 0, 0, 0.5, 0.5, 0.5]


def test_cost_estimates_are_ordered(tmp_path):
    totals = {a: estimate(cfg(tmp_path, a)).total for a in debate.PROTOCOLS}
    assert totals["rlvr"] < totals["aa"] < totals["abab"] < totals["abaaba"]


def test_report_renders_fake_runs(fake):
    import argparse

    from scripts import report

    for arm in ("aa", "abab"):
        asyncio.run(trainer.train(cfg(fake, arm), FakeService()))
    out = fake / "report"
    report.curves(argparse.Namespace(runs=[str(fake / "aa"), str(fake / "abab")], out=str(out)))
    md = (out / "report.md").read_text()
    assert "| aa | aa | 3 |" in md and (out / "mcc.png").exists() and (out / "reward_bob.png").exists()
    report.transcripts(argparse.Namespace(runs=[str(fake / "abab")], step=None, n=3, out=str(out / "t.md")))
    t = (out / "t.md").read_text()
    assert "Bob (critique)" in t and "hidden reasoning" in t and "> VERDICT:" in t


def test_word_count_counts_chinese_characters():
    assert debate.word_count("Step two divides by zero.") == 5
    assert debate.word_count("第二步除以零") == 6
    assert debate.word_count("第二步 is wrong: $x=1$") == 3 + 3
    long_zh = "这一步是错误的" * 30  # 210 characters, no spaces
    assert debate.word_penalty(long_zh, 150, 0.01) == pytest.approx(0.60)
