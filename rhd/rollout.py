"""Playing out episodes: prefix-then-diverge groups for training, single episodes for evaluation.

A training group picks one ply to train. Earlier plies are sampled once and shared (the prefix);
the chosen ply is sampled `group_size` times; each of those continues to the end of the episode
with later plies sampled once, and then the judge votes on each finished episode. Only the
chosen ply is trained, with its advantage taken relative to the rest of the group.
"""

import asyncio
from dataclasses import dataclass, field
from functools import cached_property

from rhd import debate
from rhd.grading import is_correct
from rhd.judge import parse_verdict
from rhd.sampling import Completion, ModelHandle


@dataclass
class Settings:
    votes: int = 8
    word_limit: int = 150
    word_penalty: float = 0.01  # per word over the limit; ours, the paper gives none


@dataclass
class Turn:
    kind: str
    completion: Completion
    ok: bool  # passed the format rule

    @property
    def visible(self) -> str:
        return self.completion.visible

    def record(self) -> dict:
        c = self.completion
        return {"kind": self.kind, "visible": c.visible, "reasoning_chars": len(c.reasoning),
                "n_tokens": c.n_tokens, "clean": c.clean, "ok": self.ok}


@dataclass
class Episode:
    problem: dict
    turns: list[Turn]
    verdicts: list[bool | None] = field(default_factory=list)
    judge_tokens: list[int] = field(default_factory=list)
    judge_texts: list[str] = field(default_factory=list)  # the judge's visible responses

    @cached_property
    def alice_correct(self) -> bool:
        """Ground truth, for evaluation and the RLVR reward only."""
        return self.turns[0].ok and is_correct(self.turns[0].visible, self.problem["answer"])

    @property
    def alice_share(self) -> float:
        """Share of votes saying CORRECT; a vote with no verdict counts as INCORRECT."""
        return sum(v is True for v in self.verdicts) / len(self.verdicts) if self.verdicts else 0.0

    def reward(self, ply: int, protocol: debate.Protocol, s: Settings) -> float:
        turn = self.turns[ply]
        if protocol.reward == "verifier":
            return float(self.alice_correct) if turn.ok else debate.FORMAT_FAILURE
        return debate.ply_reward(turn.kind, turn.ok, turn.visible, self.alice_share,
                                 word_limit=s.word_limit, per_word=s.word_penalty)

    def record(self, protocol: debate.Protocol, s: Settings, trained_ply: int | None = None) -> dict:
        out = {
            "problem_id": self.problem["id"],
            "problem": self.problem["problem"],
            "answer": self.problem["answer"],
            "turns": [t.record() for t in self.turns],
            "alice_correct": self.alice_correct,
            "verdicts": self.verdicts,
            "alice_share": self.alice_share,
            "judge_tokens": self.judge_tokens,
            "judge_samples": self.judge_texts[:2],  # two votes' reasoning, for reading transcripts
            "rewards": [self.reward(i, protocol, s) for i in range(len(self.turns))],
        }
        if trained_ply is not None:
            out["trained_ply"] = trained_ply
        return out


async def _sample_ply(policy: ModelHandle, kind: str, problem: dict, before: list[Turn], n: int,
                      s: Settings) -> list[Turn]:
    msgs = debate.ply_messages(kind, problem["problem"], [t.visible for t in before], s.word_limit)
    comps = await policy.sample(msgs, n)
    return [Turn(kind, c, debate.format_ok(kind, c.clean, c.visible)) for c in comps]


async def _finish(policy: ModelHandle, judge: ModelHandle | None, protocol: debate.Protocol,
                  episode: Episode, s: Settings) -> Episode:
    """Sample the remaining plies once each, then get the judge's votes."""
    if not episode.turns[0].ok:
        return episode  # Alice's solution failed the format rule: nothing for Bob or the judge
    for kind in protocol.plies[len(episode.turns):]:
        (turn,) = await _sample_ply(policy, kind, episode.problem, episode.turns, 1, s)
        episode.turns.append(turn)
    if protocol.reward == "judge" and judge is not None:
        msgs = debate.debate_judge_messages(episode.problem["problem"], [t.visible for t in episode.turns])
        comps = await judge.sample(msgs, s.votes)
        episode.verdicts = [parse_verdict(c.visible) for c in comps]
        episode.judge_tokens = [c.n_tokens for c in comps]
        episode.judge_texts = [c.visible for c in comps]
    return episode


async def play_group(policy: ModelHandle, judge: ModelHandle | None, protocol: debate.Protocol,
                     problem: dict, ply: int, group_size: int, s: Settings) -> list[Episode] | None:
    """Prefix-then-diverge at `ply`. Returns None if the shared prefix fails the format rule,
    since every episode in the group would then be decided by that one failure."""
    prefix: list[Turn] = []
    for kind in protocol.plies[:ply]:
        (turn,) = await _sample_ply(policy, kind, problem, prefix, 1, s)
        if not turn.ok:
            return None
        prefix.append(turn)
    branches = await _sample_ply(policy, protocol.plies[ply], problem, prefix, group_size, s)
    episodes = [Episode(problem, [*prefix, b]) for b in branches]
    return list(await asyncio.gather(*[_finish(policy, judge, protocol, e, s) for e in episodes]))


async def play_episode(policy: ModelHandle, judge: ModelHandle | None, protocol: debate.Protocol,
                       problem: dict, s: Settings) -> Episode:
    """One full episode, every ply sampled once (evaluation)."""
    (first,) = await _sample_ply(policy, protocol.plies[0], problem, [], 1, s)
    return await _finish(policy, judge, protocol, Episode(problem, [first]), s)
