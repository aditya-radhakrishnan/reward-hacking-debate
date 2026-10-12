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
