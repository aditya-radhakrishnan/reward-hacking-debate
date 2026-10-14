"""Thin wrapper over a Tinker sampling client: chat messages in, (hidden reasoning, visible text)
out, with token accounting for cost reporting."""

import asyncio
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

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
    forced: bool = False  # hit the token budget and was made to answer (budget forcing)
    # Kept for training: the rendered prompt and the sampled tokens with their logprobs.
    prompt: tinker.ModelInput | None = None
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


class Usage:
    def __init__(self) -> None:
        self.prefill: dict[str, int] = defaultdict(int)
        self.sampled: dict[str, int] = defaultdict(int)
        self.trained: dict[str, int] = defaultdict(int)

    def cost_usd(self) -> float:
        total = 0.0
        for column, counts in enumerate([self.prefill, self.sampled, self.trained]):
            for model, n in counts.items():
                total += n * PRICES[model][column] / 1e6
        return total

    def summary(self) -> dict:
        out = {"prefill_tokens": dict(self.prefill), "sampled_tokens": dict(self.sampled)}
        if self.trained:
            out["trained_tokens"] = dict(self.trained)
        out["cost_usd"] = round(self.cost_usd(), 2)
        return out

    def load(self, summary: dict) -> None:
        """Restore counts saved by summary(), e.g. when resuming a training run."""
        for key, counts in [("prefill_tokens", self.prefill), ("sampled_tokens", self.sampled),
                            ("trained_tokens", self.trained)]:
            for model, n in summary.get(key, {}).items():
                counts[model] += n


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

    async def sample(self, messages: list[dict], n: int, force_prefix: str | None = None) -> list[Completion]:
        """Sample n completions. With `force_prefix` and a spec that has budget_force, a completion
        that runs out of tokens is ended and continued from `force_prefix` (e.g. "VERDICT:"), so it
        still gives an answer, made with less reasoning."""
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
        seqs = [list(seq.tokens) for seq in resp.sequences]
        logprobs = [list(seq.logprobs or []) for seq in resp.sequences]
        forced = [False] * len(seqs)
        if force_prefix and self.spec.budget_force:
            await self._force(prompt, seqs, forced, force_prefix)
        out = []
        for seq_tokens, seq_logprobs, was_forced in zip(seqs, logprobs, forced):
            self.usage.sampled[self.spec.base_model] += len(seq_tokens)
            msg, term = self.renderer.parse_response(seq_tokens)
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
                    n_tokens=len(seq_tokens),
                    clean=term.is_clean,
                    forced=was_forced,
                    prompt=prompt,
                    tokens=seq_tokens,
                    logprobs=seq_logprobs,  # forced tokens have none; never train on forced output
                )
            )
        return out

    async def _force(self, prompt, seqs: list[list[int]], forced: list[bool], force_prefix: str) -> None:
        """Budget forcing for GPT-OSS: close the cut-off message, open the final channel with
        `force_prefix`, and sample a short continuation. Extends `seqs` in place."""
        if not self.spec.renderer.startswith("gpt_oss"):
            raise ValueError(f"budget forcing is only implemented for GPT-OSS, not {self.spec.renderer}")
        suffix = self.renderer.tokenizer.encode(
            "<|end|><|start|>assistant<|channel|>final<|message|>" + force_prefix, add_special_tokens=False)
        params = tinker.SamplingParams(max_tokens=32, temperature=self.spec.temperature,
                                       stop=self.renderer.get_stop_sequences())

        async def one(i: int) -> None:
            if self.renderer.parse_response(seqs[i])[1].is_clean:
                return
            head = seqs[i] + suffix
            forced_prompt = prompt.append(tinker.types.EncodedTextChunk(tokens=head))
            async with self._sem:
                resp = await self.client.sample_async(prompt=forced_prompt, num_samples=1, sampling_params=params)
            self.usage.prefill[self.spec.base_model] += forced_prompt.length
            seqs[i] = head + list(resp.sequences[0].tokens)
            forced[i] = True

        await asyncio.gather(*[one(i) for i in range(len(seqs))])
