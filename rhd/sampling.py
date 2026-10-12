"""Thin wrapper over a Tinker sampling client: chat messages in, (hidden reasoning, visible text)
out, with token accounting for cost reporting."""

import asyncio
import warnings
from collections import defaultdict
from dataclasses import dataclass

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rhd.config import PRICES, ModelSpec

# The GPT-OSS renderer warns about a second system message, but it renders ours as the Harmony
# developer message (checked in tests/test_renderers.py), which is what we want.
warnings.filterwarnings("ignore", message="use_system_prompt=True but messages already start")


@dataclass
class Completion:
    reasoning: str  # hidden chain of thought; never shown to the judge
    visible: str
    n_tokens: int
    clean: bool  # False if truncated at max_tokens or malformed


class Usage:
    def __init__(self) -> None:
        self.prefill: dict[str, int] = defaultdict(int)
        self.sampled: dict[str, int] = defaultdict(int)

    def cost_usd(self) -> float:
        total = 0.0
        for model, n in self.prefill.items():
            total += n * PRICES[model][0] / 1e6
        for model, n in self.sampled.items():
            total += n * PRICES[model][1] / 1e6
        return total

    def summary(self) -> dict:
        return {
            "prefill_tokens": dict(self.prefill),
            "sampled_tokens": dict(self.sampled),
            "cost_usd": round(self.cost_usd(), 2),
        }


class ModelHandle:
    def __init__(self, spec: ModelSpec, client: tinker.SamplingClient, usage: Usage, concurrency: int):
        self.spec = spec
        self.client = client
        self.usage = usage
        self.renderer = renderers.get_renderer(spec.renderer, get_tokenizer(spec.base_model))
        if hasattr(self.renderer, "current_date"):
            self.renderer.current_date = "2026-09-28"  # keep GPT-OSS system prompts deterministic
        self._sem = asyncio.Semaphore(concurrency)

    @classmethod
    async def create(cls, service: tinker.ServiceClient, spec: ModelSpec, usage: Usage, concurrency: int = 64):
        client = await service.create_sampling_client_async(base_model=spec.base_model)
        return cls(spec, client, usage, concurrency)

    async def sample(self, messages: list[dict], n: int) -> list[Completion]:
        prompt = self.renderer.build_generation_prompt(messages)
        params = tinker.SamplingParams(
            max_tokens=self.spec.max_tokens,
            temperature=self.spec.temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        async with self._sem:
            resp = await self.client.sample_async(prompt=prompt, num_samples=n, sampling_params=params)
        # The n samples share one prompt, so bill the prefill once (the rest are cache hits).
        self.usage.prefill[self.spec.base_model] += prompt.length
        out = []
        for seq in resp.sequences:
            self.usage.sampled[self.spec.base_model] += len(seq.tokens)
            msg, term = self.renderer.parse_response(seq.tokens)
            content = msg["content"]
            reasoning = (
                ""
                if isinstance(content, str)
                else "".join(p["thinking"] for p in content if p["type"] == "thinking")
            )
            # A truncated or malformed completion can't be split into reasoning and answer (GPT-OSS
            # cut off mid-analysis comes back as raw channel text), so treat it as having no
            # visible answer rather than leak hidden reasoning to the judge.
            visible = renderers.get_text_content(msg).strip() if term.is_clean else ""
            out.append(
                Completion(
                    reasoning=reasoning,
                    visible=visible,
                    n_tokens=len(seq.tokens),
                    clean=term.is_clean,
                )
            )
        return out
