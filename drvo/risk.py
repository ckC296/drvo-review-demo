"""Exact 1-Wasserstein collision risk and the DR safety margin."""

from dataclasses import dataclass
from typing import Any

import numpy as np


ArrayLike = Any
FloatArray = np.ndarray


@dataclass(frozen=True)
class RiskResult:
    """All quantities needed to audit one DR chance-constraint check."""

    worst_case_probability: float
    empirical_probability: float
    critical_radius: float
    safety_margin: float
    safe: bool
    sorted_distances: FloatArray


def _validated_distances(distances: ArrayLike) -> FloatArray:
    values = np.asarray(distances, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one sample distance is required")
    if np.any(values < -1e-10) or not np.all(np.isfinite(values)):
        raise ValueError("distances must be finite and nonnegative")
    return np.maximum(values, 0.0)


def worst_case_collision_risk(distances: ArrayLike, wasserstein_radius: float) -> float:
    """Evaluate Proposition 2 by fractional transportation.

    Each empirical atom has mass ``1/N``.  Moving a fraction of atom ``i``
    into the collision set costs that fraction times ``d_i/N``.  Sorting the
    distances is therefore the exact fractional-knapsack solution.
    """
    values = np.sort(_validated_distances(distances))
    return _worst_case_from_sorted(values, wasserstein_radius)


def _worst_case_from_sorted(values: FloatArray, wasserstein_radius: float) -> float:
    rho = float(wasserstein_radius)
    if rho < 0.0:
        raise ValueError("wasserstein_radius must be nonnegative")
    budget = values.size * rho
    moved_atoms = 0.0
    for distance in values:
        if distance <= 1e-12:
            moved_atoms += 1.0
        elif budget > 0.0:
            fraction = min(1.0, budget / distance)
            moved_atoms += fraction
            budget -= fraction * distance
        else:
            break
    return min(1.0, moved_atoms / values.size)


def critical_wasserstein_radius(distances: ArrayLike, risk_level: float) -> float:
    """Return the minimum budget ``rho_alpha(v)`` in manuscript Eq. (95)."""
    values = np.sort(_validated_distances(distances))
    return _critical_radius_from_sorted(values, risk_level)


def _critical_radius_from_sorted(values: FloatArray, risk_level: float) -> float:
    alpha = float(risk_level)
    if not 0.0 < alpha < 1.0:
        raise ValueError("risk_level must lie strictly between zero and one")
    scaled = values.size * alpha
    k_alpha = int(np.floor(scaled + 1e-14))
    gamma_alpha = scaled - k_alpha
    total = float(np.sum(values[:k_alpha]))
    if gamma_alpha > 1e-14:
        total += gamma_alpha * float(values[k_alpha])
    return total / values.size


def dr_safety_margin(
    distances: ArrayLike,
    risk_level: float,
    wasserstein_radius: float,
    tolerance: float = 1e-10,
) -> RiskResult:
    """Evaluate manuscript Eqs. (96)--(102) and return diagnostics."""
    values = np.sort(_validated_distances(distances))
    alpha = float(risk_level)
    rho = float(wasserstein_radius)
    if rho < 0.0:
        raise ValueError("wasserstein_radius must be nonnegative")
    empirical = float(np.count_nonzero(values <= tolerance) / values.size)
    if not 0.0 < alpha < 1.0:
        raise ValueError("risk_level must lie strictly between zero and one")
    critical = _critical_radius_from_sorted(values, alpha)
    margin = critical - rho
    worst_case = _worst_case_from_sorted(values, rho)
    safe = empirical <= alpha + tolerance and margin >= -tolerance
    # The probability form is retained as an independent runtime diagnostic.
    if safe != (worst_case <= alpha + 5.0 * tolerance):
        raise RuntimeError("risk and critical-radius safety tests disagree")
    return RiskResult(
        worst_case_probability=worst_case,
        empirical_probability=empirical,
        critical_radius=critical,
        safety_margin=margin,
        safe=safe,
        sorted_distances=values,
    )
