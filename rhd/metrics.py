import math


def mcc(preds: list[bool], labels: list[bool]) -> float:
    """Matthews correlation coefficient; 0.0 when undefined (a row or column of the confusion
    matrix is empty)."""
    tp = sum(p and y for p, y in zip(preds, labels))
    tn = sum(not p and not y for p, y in zip(preds, labels))
    fp = sum(p and not y for p, y in zip(preds, labels))
    fn = sum(not p and y for p, y in zip(preds, labels))
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Probability a random correct solution scores higher than a random incorrect one
    (ties count half). NaN if either class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
