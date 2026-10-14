"""Checks prompt rendering for the pilot's models (downloads tokenizers; no API key)."""

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rhd.config import JUDGES, POLICY
from rhd.judge import judge_messages, solver_messages


def _render(spec, messages):
    r = renderers.get_renderer(spec.renderer, get_tokenizer(spec.base_model))
    if hasattr(r, "current_date"):
        r.current_date = "2026-09-28"
    prompt = r.build_generation_prompt(messages)
    return r, prompt, r.tokenizer.decode(prompt.to_ints())


def test_policy_prompt_has_medium_effort_and_problem():
    _, prompt, text = _render(POLICY, solver_messages("What is $1+1$?"))
    assert "Reasoning: medium" in text and "What is $1+1$?" in text
    assert prompt.length > 0


def test_qwen_judge_disables_thinking():
    _, _, text = _render(JUDGES["qwen3-8b-nothink"], judge_messages("P", "S \\boxed{1}"))
    assert text.rstrip().endswith("</think>")


def test_gptoss_parse_separates_hidden_reasoning():
    r, _, _ = _render(POLICY, solver_messages("x"))
    sampled = "<|channel|>analysis<|message|>secret work<|end|><|start|>assistant<|channel|>final<|message|>\\boxed{2}<|return|>"
    msg, term = r.parse_response(r.tokenizer.encode(sampled, add_special_tokens=False))
    assert renderers.get_text_content(msg).strip() == "\\boxed{2}"
    assert any(p["type"] == "thinking" and "secret" in p["thinking"] for p in msg["content"])


def test_critique_prompt_renders_alice_visible_solution_only():
    from rhd import debate

    msgs = debate.ply_messages("critique", "What is $1+1$?", ["So \\boxed{2}"], word_limit=150)
    _, _, text = _render(POLICY, msgs)
    assert "Reasoning: medium" in text and "So \\boxed{2}" in text and "under 150 words" in text


def test_budget_forcing_turns_a_cut_off_vote_into_a_verdict():
    """A judge vote that hits max_tokens mid-reasoning is closed and continued from "VERDICT:"."""
    import asyncio
    from types import SimpleNamespace

    from rhd.config import JUDGES
    from rhd.judge import FORCE_VERDICT, parse_verdict
    from rhd.sampling import ModelHandle, Usage

    spec = JUDGES["gptoss20b-medium-8k-forced"]
    tok = get_tokenizer(spec.base_model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    finished = enc("<|channel|>analysis<|message|>ok<|end|><|start|>assistant<|channel|>final<|message|>VERDICT: INCORRECT<|return|>")
    cut_off = enc("<|channel|>analysis<|message|>still checking step three and")

    class FakeClient:
        def __init__(self):
            self.prompts = []

        async def sample_async(self, prompt, num_samples, sampling_params):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                seqs = [finished, cut_off]
            else:
                seqs = [enc(" CORRECT<|return|>")]
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=s, logprobs=None) for s in seqs])

    client = FakeClient()
    handle = ModelHandle(spec, client, Usage(), concurrency=4)
    done, forced = asyncio.run(handle.sample(judge_messages("P", "S \\boxed{1}"), 2, FORCE_VERDICT))
    assert not done.forced and parse_verdict(done.visible) is False
    assert forced.forced and forced.clean and parse_verdict(forced.visible) is True
    assert "still checking" in forced.reasoning and "still checking" not in forced.visible
    assert len(client.prompts) == 2  # only the cut-off vote was continued
    tail = tok.decode(client.prompts[1].to_ints()[-12:])
    assert tail.endswith("<|channel|>final<|message|>VERDICT:")

    # Without a force prefix (or for a spec without budget_force), a cut-off stays cut off.
    client2 = FakeClient()
    _, cut = asyncio.run(ModelHandle(spec, client2, Usage(), 4).sample(judge_messages("P", "S"), 2))
    assert not cut.forced and not cut.clean and cut.visible == ""
