"""Final-answer checking against the reference answer (used for evaluation only, never as reward
in the LLM-judge protocols)."""

from math_verify import parse, verify
from tinker_cookbook.recipes.math_rl.math_env import safe_grade
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


def extract_answer(visible_text: str) -> str | None:
    try:
        return extract_boxed(visible_text)
    except ValueError:
        return None


def _dollars(s: str) -> str:
    return s if s.startswith("$") and s.endswith("$") else f"${s}$"


def math_verify_equal(given: str, reference: str) -> bool:
    """math-verify with its own signal-based timeouts, so this must run on the main thread. The
    cookbook's safe_grade wraps math-verify in a worker thread, where it always errors out."""
    try:
        return verify(parse(_dollars(reference)), parse(_dollars(given)))  # gold first
    except Exception:
        return False


def is_correct(visible_text: str, reference: str) -> bool:
    given = extract_answer(visible_text)
    if given is None:
        return False
    # Fall back to the cookbook's sympy grader, which handles some formats math-verify rejects.
    return math_verify_equal(given, reference) or safe_grade(given, reference, grader="sympy")
