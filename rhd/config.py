"""Model choices and Tinker list prices (USD per million tokens), as of 2026-09-28."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # key used on the command line and in output files
    base_model: str  # Tinker base model id
    renderer: str  # tinker_cookbook renderer name
    max_tokens: int
    temperature: float = 1.0


# Prices: (prefill, sample, train) per 1M tokens. Cached prefill is 20% of prefill.
PRICES = {
    "openai/gpt-oss-20b": (0.18, 0.45, 0.396),
    "openai/gpt-oss-120b": (0.33, 0.84, 0.737),
    "Qwen/Qwen3-8B": (0.195, 0.60, 0.44),
}

POLICIES = {
    p.name: p
    for p in [
        ModelSpec("gptoss120b-medium", "openai/gpt-oss-120b", "gpt_oss_medium_reasoning", max_tokens=12_000),
        ModelSpec("gptoss20b-medium", "openai/gpt-oss-20b", "gpt_oss_medium_reasoning", max_tokens=12_000),
    ]
}
# Chosen after the pilot (2026-09-30): no cheap judge reached step-0 MCC 0.4 against the 20B
# policy, so the policy moved up to 120B with GPT-OSS-20B (medium effort) as the judge.
POLICY = POLICIES["gptoss120b-medium"]

# Judge candidates for the pilot, in order of preference (see plan section 4c).
JUDGES = {
    j.name: j
    for j in [
        ModelSpec("qwen3-8b-nothink", "Qwen/Qwen3-8B", "qwen3_disable_thinking", max_tokens=1_024),
        ModelSpec("gptoss20b-low", "openai/gpt-oss-20b", "gpt_oss_low_reasoning", max_tokens=4_096),
        ModelSpec("qwen3-8b-think", "Qwen/Qwen3-8B", "qwen3", max_tokens=4_096),
        ModelSpec("gptoss20b-medium", "openai/gpt-oss-20b", "gpt_oss_medium_reasoning", max_tokens=8_192),
        # Same judge with a cap near the 32K context, to see how long it reasons when not cut off.
        # The paper reports no judge token limit.
        ModelSpec("gptoss20b-medium-24k", "openai/gpt-oss-20b", "gpt_oss_medium_reasoning", max_tokens=24_000),
    ]
}

JUDGE_VOTES = 8
# Chosen 2026-10-01 (plan section 4e): medium effort with a 24k cap, so only ~1% of votes are cut off.
JUDGE = JUDGES["gptoss20b-medium-24k"]
