"""Paper Eq. 8: shift emphasis from reference proximity to predicted return.

n counts learner updates from the start of warm-up.
"""

from __future__ import annotations


def alpha(n: int, start: int = 20000, span: int = 50000) -> float:
    return min(max((n - start) / span, 0.0), 1.0)


def lambda_ref(n: int, start: int = 20000, span: int = 50000) -> float:
    return 7.0 - 4.5 * alpha(n, start, span)


def lambda_q(n: int, start: int = 20000, span: int = 50000) -> float:
    return 0.05 + 0.40 * alpha(n, start, span)
