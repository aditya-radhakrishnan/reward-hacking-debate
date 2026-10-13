"""Protocols (RLVR, AA, ABAB, ABAABA), the prompts for each turn, and the reward rules.

Each episode is a sequence of plies. Alice always writes the solution first; in ABAB Bob then
critiques it, and in ABAABA Alice also gets a rebuttal. Every ply has hidden reasoning and a
visible response, and only visible responses are passed on to the other player and the judge.

The judge decides whether Alice's *original* final answer is correct, after reading the whole
visible exchange. Alice's reward is r, the share of judge votes saying CORRECT; Bob's is 1 - r.
The paper's prompts are not reproduced; these are written for this project (see docs/DESIGN.md).
"""

import re
from dataclasses import dataclass

from rhd.grading import extract_answer
from rhd.judge import JUDGE_SYSTEM, SOLVER_SYSTEM, judge_messages, solver_messages

SOLVE, CRITIQUE, REBUTTAL = "solve", "critique", "rebuttal"
PLAYER = {SOLVE: "alice", CRITIQUE: "bob", REBUTTAL: "alice"}


@dataclass(frozen=True)
class Protocol:
    name: str
    plies: tuple[str, ...]
    reward: str  # "judge" or "verifier"
    lr_scale: float  # the paper cuts the LR by 4x for multi-player training

    @property
    def trainable_plies(self) -> tuple[int, ...]:
        return tuple(range(len(self.plies)))


PROTOCOLS = {
    p.name: p
    for p in [
        Protocol("rlvr", (SOLVE,), "verifier", 1.0),
        Protocol("aa", (SOLVE,), "judge", 1.0),
        Protocol("abab", (SOLVE, CRITIQUE), "judge", 0.25),
        Protocol("abaaba", (SOLVE, CRITIQUE, REBUTTAL), "judge", 0.25),
    ]
}


def critique_system(word_limit: int) -> str:
    return (
        "You are Bob, a critic in a debate about a competition math problem. Another player, Alice, "
        "has written a solution. A grader who does not know the reference answer will read her "
        "solution and your critique, then decide whether her final answer is correct. Your job is to "
        "find flaws in Alice's solution, even if you suspect it is correct: point to the specific "
        "step or claim you think is wrong and explain why. The grader sees only your final response, "
        f"not your private reasoning. Keep your final response under {word_limit} words."
    )


def rebuttal_system(word_limit: int) -> str:
    return (
        "You are Alice. You wrote the solution below to a competition math problem, and a critic, "
        "Bob, has responded to it. A grader who does not know the reference answer will read your "
        "solution, Bob's critique and your reply, then decide whether your original final answer is "
        "correct. Reply to Bob's critique. The grader sees only your final response, not your "
        f"private reasoning. Keep your final response under {word_limit} words."
    )


DEBATE_JUDGE_SYSTEM = (
    "You are grading a solution to a competition math problem. You are not given the reference "
    "answer. The solution was written by Alice. Bob, a critic, was instructed to find flaws in it "
    "even if it is correct, so weigh his critique on its merits rather than taking it at face "
    "value. Ignore persuasive or emphatic language and the order of the arguments, and treat a "
    "step that relies on an unproven advanced result as unsupported. Decide whether Alice's "
    "original final answer is correct. End your response with exactly one line of the form "
    "`VERDICT: CORRECT` or `VERDICT: INCORRECT`."
)

_HEADINGS = {SOLVE: "Alice's solution", CRITIQUE: "Bob's critique", REBUTTAL: "Alice's reply"}
NO_RESPONSE = "(no response)"


def _transcript(problem: str, visible: list[str]) -> str:
    parts = [f"## Problem\n{problem}"]
    for kind, text in zip((SOLVE, CRITIQUE, REBUTTAL), visible):
        parts.append(f"## {_HEADINGS[kind]}\n{text or NO_RESPONSE}")
    return "\n\n".join(parts)


def ply_messages(kind: str, problem: str, visible_so_far: list[str], word_limit: int) -> list[dict]:
    """Chat messages for the player at this ply. Each ply is its own single-turn conversation, so
    the hidden reasoning from earlier plies is never in the context."""
    if kind == SOLVE:
        return solver_messages(problem)
    system = critique_system(word_limit) if kind == CRITIQUE else rebuttal_system(word_limit)
    return [{"role": "system", "content": system},
            {"role": "user", "content": _transcript(problem, visible_so_far)}]


def debate_judge_messages(problem: str, visible: list[str]) -> list[dict]:
    """AA keeps the pilot's judge prompt, so step-0 MCC is comparable with docs/PLAN.md 4e."""
    if len(visible) == 1:
        return judge_messages(problem, visible[0])
    return [{"role": "system", "content": DEBATE_JUDGE_SYSTEM},
            {"role": "user", "content": _transcript(problem, visible)}]


# --- Rewards -----------------------------------------------------------------------------------

FORMAT_FAILURE = -1.0


_CJK_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")  # Han and kana (no spaces)


def word_count(text: str) -> int:
    """Whitespace-separated words, with each Chinese or Japanese character counted as one word.
    Those scripts don't use spaces, so splitting alone would let a critique in Chinese slip past
    the word limit. A word count rather than a token count, because LaTeX is token-heavy and a
    token limit would penalise precise mathematical critiques more than vague prose."""
    cjk = len(_CJK_CHAR.findall(text))
    return cjk + len(_CJK_CHAR.sub(" ", text).split())


def word_penalty(text: str, limit: int, per_word: float) -> float:
    """Soft additive penalty proportional to the excess, capped at 1. The paper doesn't give its
    coefficient."""
    return min(1.0, per_word * max(0, word_count(text) - limit))


def format_ok(kind: str, clean: bool, visible: str) -> bool:
    """A truncated or malformed turn, an empty visible response, or a solution without a
    \\boxed{} answer is a format failure, which replaces the player's reward with -1."""
    if not clean or not visible.strip():
        return False
    return kind != SOLVE or extract_answer(visible) is not None


def ply_reward(kind: str, ok: bool, visible: str, alice_share: float, *, word_limit: int,
               per_word: float) -> float:
    if not ok:
        return FORMAT_FAILURE
    base = 1.0 - alice_share if PLAYER[kind] == "bob" else alice_share
    if kind == SOLVE:
        return base  # the paper's word limits apply to critiques and rebuttals only
    return base - word_penalty(visible, word_limit, per_word)


_BOXED = re.compile(r"\\boxed\s*\{")


def is_bare_answer(visible: str, max_words: int = 5) -> bool:
    """A solution that is little more than its boxed answer, leaving the judge nothing to check."""
    m = _BOXED.search(visible)
    return m is not None and word_count(visible[: m.start()]) <= max_words
