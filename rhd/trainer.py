"""The RL training loop for all four arms (RLVR, AA, ABAB, ABAABA). See docs/DESIGN.md.

Synchronous on-policy training: each step samples with the current weights, scores the
episodes, and takes one optimizer step on the trained plies with the `importance_sampling` loss.
Everything a run produces goes under its run directory:

  config.json        the settings the run was started with
  metrics.jsonl      one row per training step
  eval.jsonl         one row per evaluation (every `eval_every` steps, and at the start and end)
  checkpoints.jsonl  Tinker state paths, saved every `save_every` steps and after each
                     evaluation, for resuming
  usage.json         tokens and dollars spent so far
  transcripts/       sampled training episodes and every evaluation episode, as JSONL
"""

import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tinker
import torch
from tinker import types

from rhd import data, debate
from rhd.config import JUDGE, JUDGES, POLICIES, POLICY
from rhd.metrics import auroc, mcc
from rhd.rollout import Episode, Settings, play_episode, play_group
from rhd.sampling import Completion, ModelHandle, Usage


@dataclass
class RunConfig:
    arm: str
    run_dir: str
    steps: int = 200
    groups_per_step: int = 32
    group_size: int = 8
    lr: float = 4e-5  # base LR; the protocol's lr_scale (1/4 for debate) is applied on top
    lora_rank: int = 32
    votes: int = 8
    word_limit: int = 150
    word_penalty: float = 0.01
    eval_every: int = 20
    save_every: int = 5
    eval_problems: int = 150
    eval_samples: int = 2
    transcripts_per_step: int = 8
    max_usd: float = 0.0
    seed: int = 0
    english_only: bool = True
    data_path: str = "runs/data_dapo/filtered.jsonl"
    policy: str = POLICY.name
    judge: str = JUDGE.name
    concurrency: int = 64

    @property
    def settings(self) -> Settings:
        return Settings(votes=self.votes, word_limit=self.word_limit, word_penalty=self.word_penalty)


# Settings that must not change when a run is resumed.
_FIXED_ON_RESUME = ["arm", "groups_per_step", "group_size", "lr", "lora_rank", "votes", "word_limit",
                    "word_penalty", "seed", "english_only", "data_path", "policy", "judge"]


def batch_problems(train: list[dict], step: int, n: int, seed: int) -> list[dict]:
    """The problems for one step: epochs over the train set, each in its own shuffled order, so a
    resumed run sees exactly the batches it would have seen."""
    out, start = [], step * n
    for pos in range(start, start + n):
        epoch, i = divmod(pos, len(train))
        order = list(range(len(train)))
        random.Random(f"{seed}:epoch:{epoch}").shuffle(order)
        out.append(train[order[i]])
    return out


def make_datum(c: Completion, advantage: float) -> types.Datum:
    """Prompt positions get advantage 0, so only the sampled tokens of the trained ply learn."""
    ob_len = c.prompt.length - 1
    model_input = c.prompt.append(types.EncodedTextChunk(tokens=c.tokens[:-1]))
    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor([0] * ob_len + c.tokens)),
            "logprobs": types.TensorData.from_torch(torch.tensor([0.0] * ob_len + c.logprobs)),
            "advantages": types.TensorData.from_torch(
                torch.tensor([0.0] * ob_len + [advantage] * len(c.tokens))),
        },
    )


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(float(statistics.mean(xs)), 4) if xs else None


def episode_stats(episodes: list[Episode], protocol: debate.Protocol) -> dict:
    """Per-ply lengths and format failures, plus the bare-answer rate of Alice's solutions."""
    out = {}
    for j, kind in enumerate(protocol.plies):
        turns = [e.turns[j] for e in episodes if len(e.turns) > j]
        out[f"{kind}/words"] = _mean(debate.word_count(t.visible) for t in turns)
        out[f"{kind}/tokens"] = _mean(t.completion.n_tokens for t in turns)
        out[f"{kind}/format_fail"] = _mean(not t.ok for t in turns)
    out["bare_answer_rate"] = _mean(debate.is_bare_answer(e.turns[0].visible) for e in episodes if e.turns[0].ok)
    out["judge_tokens"] = _mean(t for e in episodes for t in e.judge_tokens)
    return out


def judge_metrics(episodes: list[Episode]) -> dict:
    """The judge as a classifier of whether Alice's answer is correct. Episodes the judge never
    saw (Alice's solution failed the format rule) are left out, as in the pilot."""
    judged = [e for e in episodes if e.verdicts]
    labels = [e.alice_correct for e in judged]
    shares = [e.alice_share for e in judged]
    votes = [v for e in judged for v in e.verdicts]
    return {
        "judged": len(judged),
        "mcc": round(mcc([s >= 0.5 for s in shares], labels), 4),
        "auroc": round(auroc(shares, labels), 4),
        "alice_share": _mean(shares),
        "no_verdict_rate": _mean(v is None for v in votes),
    }


class Run:
    def __init__(self, cfg: RunConfig, service):
        self.cfg, self.service = cfg, service
        self.dir = Path(cfg.run_dir)
        self.protocol = debate.PROTOCOLS[cfg.arm]
        # RLVR trains without a judge but is measured with the AA judge, like the paper's roofline.
        self.eval_protocol = debate.PROTOCOLS["aa"] if cfg.arm == "rlvr" else self.protocol
        self.policy_spec, self.judge_spec = POLICIES[cfg.policy], JUDGES[cfg.judge]
        self.usage = Usage()
        rows = data.read_jsonl(Path(cfg.data_path))
        self.train, self.val = data.train_val_split(rows, seed=0, english_only=cfg.english_only)
        self.lr = cfg.lr * self.protocol.lr_scale
        self.last_step_usd: float | None = None
        self.last_eval_usd: float | None = None
        self.last_ckpt_step: int | None = None

    # --- files ---------------------------------------------------------------------------------

    def _append(self, name: str, row: dict) -> None:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _write_config(self) -> None:
        path = self.dir / "config.json"
        now = {**asdict(self.cfg), "effective_lr": self.lr, "train_problems": len(self.train),
               "val_problems": len(self.val)}
        if path.exists():
            before = json.loads(path.read_text())
            changed = [k for k in _FIXED_ON_RESUME if before.get(k) != now.get(k)]
            if changed:
                raise SystemExit(f"{path} was started with different {changed}; use a new --run-dir")
        path.write_text(json.dumps(now, indent=2) + "\n")

    def _save_usage(self) -> None:
        (self.dir / "usage.json").write_text(json.dumps(self.usage.summary(), indent=2) + "\n")

    def _evaluated_steps(self) -> set[int]:
        return {r["step"] for r in data.read_jsonl(self.dir / "eval.jsonl")}

    # --- Tinker --------------------------------------------------------------------------------

    async def _training_client(self) -> tuple[object, int]:
        ckpts = data.read_jsonl(self.dir / "checkpoints.jsonl")
        if ckpts:
            last = ckpts[-1]
            self._rewind(last["step"])
            print(f"resuming from step {last['step']} ({last['state_path']}), "
                  f"${self.usage.cost_usd():.2f} spent so far")
            tc = await self.service.create_training_client_from_state_with_optimizer_async(last["state_path"])
            self.last_ckpt_step = last["step"]
            return tc, last["step"]
        tc = await self.service.create_lora_training_client_async(
            base_model=self.policy_spec.base_model, rank=self.cfg.lora_rank)
        return tc, 0

    def _rewind(self, step: int) -> None:
        """Drop logs from steps after the checkpoint (they are redone), but keep counting their
        spend: usage.json, not the checkpoint, has what was actually spent."""
        usage_path = self.dir / "usage.json"
        self.usage.load(json.loads(usage_path.read_text()) if usage_path.exists() else {})
        for name in ["metrics.jsonl", "eval.jsonl"]:
            rows = data.read_jsonl(self.dir / name)
            if rows:
                data.write_jsonl(self.dir / name, [r for r in rows if r["step"] <= step])
        for kind in ["train", "eval"]:
            for f in (self.dir / "transcripts" / kind).glob("step*.jsonl"):
                if int(f.stem[4:]) > step:
                    f.unlink()

    async def _checkpoint(self, tc, step: int) -> None:
        if self.last_ckpt_step == step:
            return
        name = f"{self.dir.name}-step{step:04d}"
        fut = await tc.save_state_async(name)
        path = (await fut.result_async()).path
        self._save_usage()
        self._append("checkpoints.jsonl", {"step": step, "state_path": path, "usage": self.usage.summary(),
                                           "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.last_ckpt_step = step

    # --- the loop ------------------------------------------------------------------------------

    def _over_budget(self, next_cost: float | None) -> bool:
        spent = self.usage.cost_usd()
        return spent + (next_cost or 0.0) > self.cfg.max_usd

    async def run(self) -> None:
        from rhd.costs import estimate

        cfg = self.cfg
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_config()
        est = estimate(cfg)
        tc, step = await self._training_client()
        judge = await ModelHandle.create(self.service, self.judge_spec, self.usage, cfg.concurrency)
        policy: ModelHandle | None = None
        adam = types.AdamParams(learning_rate=self.lr, beta1=0.9, beta2=0.95, eps=1e-8)
        print(f"{cfg.arm}: {len(self.train)} train / {len(self.val)} val problems, lr {self.lr:g}, "
              f"steps {step}->{cfg.steps}, cap ${cfg.max_usd:.0f}")

        while True:
            sampler = await tc.save_weights_and_get_sampling_client_async()
            if policy is None:
                policy = ModelHandle(self.policy_spec, sampler, self.usage, cfg.concurrency)
            policy.client = sampler

            due = step % cfg.eval_every == 0 or step == cfg.steps
            if due and step not in self._evaluated_steps():
                if self._over_budget(self.last_eval_usd or est.per_eval):
                    print(f"stopping before the step-{step} eval: it would pass the ${cfg.max_usd:.0f} cap")
                    await self._checkpoint(tc, step)
                    break
                await self.evaluate(policy, judge, step)
                await self._checkpoint(tc, step)
            if step >= cfg.steps:
                break
            if self._over_budget(self.last_step_usd or est.per_step):
                print(f"stopping before step {step + 1}: it would pass the ${cfg.max_usd:.0f} cap")
                await self._checkpoint(tc, step)
                break
            await self.train_step(tc, policy, judge, adam, step)
            step += 1
            if step % cfg.save_every == 0:
                await self._checkpoint(tc, step)

        self._save_usage()
        print(f"done at step {step}; total spend ${self.usage.cost_usd():.2f}")

    async def train_step(self, tc, policy: ModelHandle, judge: ModelHandle, adam, step: int) -> None:
        cfg, s, t0, usd0 = self.cfg, self.cfg.settings, time.time(), self.usage.cost_usd()
        rng = random.Random(f"{cfg.seed}:step:{step}")
        problems = batch_problems(self.train, step, cfg.groups_per_step, cfg.seed)
        plies = [rng.choice(self.protocol.trainable_plies) for _ in problems]
        uses_judge = self.protocol.reward == "judge"
        groups = await asyncio.gather(*[
            play_group(policy, judge if uses_judge else None, self.protocol, p, ply, cfg.group_size, s)
            for p, ply in zip(problems, plies)
        ])

        datums, rewards_by_player, episodes = [], {"alice": [], "bob": []}, []
        skipped_prefix = skipped_flat = 0
        transcripts = []
        for group, ply in zip(groups, plies):
            if group is None:
                skipped_prefix += 1
                continue
            rewards = [e.reward(ply, self.protocol, s) for e in group]
            rewards_by_player[debate.PLAYER[self.protocol.plies[ply]]] += rewards
            episodes += group
            if len(transcripts) < cfg.transcripts_per_step:
                transcripts.append(group[0].record(self.protocol, s, trained_ply=ply))
            mean = statistics.mean(rewards)
            if all(r == mean for r in rewards):
                skipped_flat += 1
                continue
            datums += [make_datum(e.turns[ply].completion, r - mean) for e, r in zip(group, rewards)]

        if datums:
            fb = await tc.forward_backward_async(datums, "importance_sampling")
            opt = await tc.optim_step_async(adam)
            await fb.result_async()
            await opt.result_async()
            self.usage.trained[self.policy_spec.base_model] += sum(d.model_input.length for d in datums)

        self.last_step_usd = self.usage.cost_usd() - usd0
        row = {
            "step": step + 1,
            "reward/alice": _mean(rewards_by_player["alice"]),
            "reward/bob": _mean(rewards_by_player["bob"]),
            "train_accuracy": _mean(e.alice_correct for e in episodes),
            "alice_share": _mean(e.alice_share for e in episodes) if uses_judge else None,
            **episode_stats(episodes, self.protocol),
            "groups_trained": len(groups) - skipped_prefix - skipped_flat,
            "groups_skipped_prefix": skipped_prefix,
            "groups_skipped_flat": skipped_flat,
            "datums": len(datums),
            "lr": self.lr,
            "usd_step": round(self.last_step_usd, 3),
            "usd_total": round(self.usage.cost_usd(), 2),
            "seconds": round(time.time() - t0, 1),
        }
        self._append("metrics.jsonl", row)
        for t in transcripts:
            self._append(f"transcripts/train/step{step + 1:04d}.jsonl", t)
        self._save_usage()
        print(f"step {step + 1}: reward {row['reward/alice']} (alice) {row['reward/bob']} (bob), "
              f"acc {row['train_accuracy']}, {len(datums)} datums, ${row['usd_step']:.2f}, {row['seconds']}s")

    async def evaluate(self, policy: ModelHandle, judge: ModelHandle, step: int) -> None:
        cfg, usd0, t0 = self.cfg, self.usage.cost_usd(), time.time()
        problems = self.val[: cfg.eval_problems]
        episodes = await asyncio.gather(*[
            play_episode(policy, judge, self.eval_protocol, p, cfg.settings)
            for p in problems for _ in range(cfg.eval_samples)
        ])
        self.last_eval_usd = self.usage.cost_usd() - usd0
        row = {
            "step": step,
            "accuracy": _mean(e.alice_correct for e in episodes),
            **judge_metrics(episodes),
            **episode_stats(episodes, self.eval_protocol),
            "episodes": len(episodes),
            "usd_eval": round(self.last_eval_usd, 3),
            "usd_total": round(self.usage.cost_usd(), 2),
            "seconds": round(time.time() - t0, 1),
        }
        self._append("eval.jsonl", row)
        for e in episodes:
            self._append(f"transcripts/eval/step{step:04d}.jsonl", e.record(self.eval_protocol, cfg.settings))
        print(f"eval @ {step}: accuracy {row['accuracy']}, judge MCC {row['mcc']}, "
              f"alice share {row['alice_share']}, ${row['usd_eval']:.2f}")


async def train(cfg: RunConfig, service=None) -> None:
    await Run(cfg, service or tinker.ServiceClient()).run()
