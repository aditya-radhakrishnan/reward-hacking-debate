"""Tests that need no Tinker API key."""

import math

from rhd.data import strip_dapo_prompt
from rhd.grading import is_correct
from rhd.judge import parse_verdict
from rhd.metrics import auroc, mcc


def test_mcc():
    assert mcc([True, False, True, False], [True, False, True, False]) == 1.0
    assert mcc([False, True], [True, False]) == -1.0
    assert mcc([True, True], [True, False]) == 0.0  # undefined -> 0


def test_auroc():
    assert auroc([0.9, 0.1], [True, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(auroc([0.5], [True]))


def test_parse_verdict_uses_last_match():
    assert parse_verdict("Format is VERDICT: CORRECT ... hmm\nVERDICT: INCORRECT") is False
    assert parse_verdict("**VERDICT:** CORRECT".replace("**VERDICT:**", "VERDICT: **")) is True
    assert parse_verdict("looks right") is None


def test_strip_dapo_prompt():
    raw = ('Solve the following math problem step by step. The last line of your response should be of '
           'the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
           'What is $1+1$?\n\nRemember to put your answer on its own line after "Answer:".')
    assert strip_dapo_prompt(raw) == "What is $1+1$?"


def test_is_correct():
    assert is_correct("so the answer is \\boxed{\\frac{1}{2}}", "0.5")
    assert is_correct("\\boxed{34}", "34")
    assert not is_correct("\\boxed{35}", "34")
    assert not is_correct("no boxed answer here", "34")


def test_math_verify_runs_on_main_thread():
    # Regression: the cookbook's threaded wrapper made math-verify fail on every call.
    from rhd.grading import math_verify_equal

    assert math_verify_equal("n^n", "n^n")
    assert math_verify_equal("\\dfrac{3}{6}", "\\frac{1}{2}")
    assert not math_verify_equal("n^2", "n^n")


def test_checkable_reference():
    from rhd.data import is_checkable_reference

    for ok in ["34", "-\\frac{9}{2}", "n^n", "5 - 4\\sqrt{2}", "\\dfrac {\\pi}{3}", "2^{r+1}", "(1, 0, -1)"]:
        assert is_checkable_reference(ok), ok
    for bad in ["\\text{no}", "All powers of 2", "p \\leq 1/4", "[3,7] \\cup [8,9)", "(2, 3, 5), (2, 5, 3)", ""]:
        assert not is_checkable_reference(bad), bad
