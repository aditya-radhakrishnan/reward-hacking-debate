"""Cost estimates for a training run, from token counts measured in the pilot (docs/PLAN.md 4e).

These are rough (about +/-30%): the critique and rebuttal lengths are guesses until a debate run
measures them, and response lengths drift during training.
"""

from dataclasses import dataclass

from rhd.config import PRICES
from rhd.debate import PROTOCOLS

# Mean tokens per call. Policy and judge figures were measured in the pilot; the critique and
# rebuttal reuse the solution figure (hidden reasoning dominates either way).
POLICY_COMPLETION = 4_300
POLICY_PROMPT = {"solve": 300, "critique": 1_000, "rebuttal": 1_300}
JUDGE_COMPLETION = 5_400  # GPT-OSS-20B medium at a 24k cap
JUDGE_PROMPT = {1: 800, 2: 1_100, 3: 1_400}  # by number of plies the judge reads


@dataclass
class Estimate:
    per_step: float
    training: float
    per_eval: float
    evaluation: float

    @property
    def total(self) -> float:
        return self.training + self.evaluation

    def lines(self) -> list[str]:
        return [
            f"  per training step  ${self.per_step:8.2f}",
            f"  training           ${self.training:8.2f}",
            f"  per evaluation     ${self.per_eval:8.2f}",
            f"  evaluation         ${self.evaluation:8.2f}",
            f"  total              ${self.total:8.2f}  (rough, +/-30%; add ~25% for reruns)",
        ]


def _sample(model: str, prompt: int, completion: int, calls: float, completions: float) -> float:
    prefill, sample, _ = PRICES[model]
    return (calls * prompt * prefill + completions * completion * sample) / 1e6


def estimate(cfg) -> Estimate:
    """`cfg` is a trainer.RunConfig."""
    from rhd.config import JUDGES, POLICIES

    protocol = PROTOCOLS[cfg.arm]
    policy, judge = POLICIES[cfg.policy].base_model, JUDGES[cfg.judge].base_model
    n_plies, g = len(protocol.plies), cfg.group_size
    uses_judge = protocol.reward == "judge"

    # One group, averaged over which ply is trained.
    group = 0.0
    for k in protocol.trainable_plies:
        for j, kind in enumerate(protocol.plies):
            n = 1 if j < k else g  # prefix plies once, the chosen ply and later ones once per branch
            calls = 1 if j <= k else g
            group += _sample(policy, POLICY_PROMPT[kind], POLICY_COMPLETION, calls, n)
        group += g * (POLICY_PROMPT[protocol.plies[k]] + POLICY_COMPLETION) * PRICES[policy][2] / 1e6
        if uses_judge:
            group += _sample(judge, JUDGE_PROMPT[n_plies], JUDGE_COMPLETION, g, g * cfg.votes)
    group /= len(protocol.trainable_plies)
    per_step = group * cfg.groups_per_step

    # Evaluation always uses a judge (RLVR is measured with the AA judge).
    episodes = cfg.eval_problems * cfg.eval_samples
    per_eval = sum(_sample(policy, POLICY_PROMPT[k], POLICY_COMPLETION, episodes, episodes) for k in protocol.plies)
    per_eval += _sample(judge, JUDGE_PROMPT[n_plies], JUDGE_COMPLETION, episodes, episodes * cfg.votes)
    n_evals = cfg.steps // cfg.eval_every + 1 + (cfg.steps % cfg.eval_every != 0)

    return Estimate(per_step, per_step * cfg.steps, per_eval, per_eval * n_evals)
