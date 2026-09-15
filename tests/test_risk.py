import numpy as np
from scipy.optimize import linprog

from drvo.risk import (
    critical_wasserstein_radius,
    dr_safety_margin,
    worst_case_collision_risk,
)


def risk_by_linear_program(distances: np.ndarray, rho: float) -> float:
    """Maximize transported mass with scipy, independently of the sorting formula."""
    n = len(distances)
    result = linprog(
        c=-np.ones(n) / n,
        A_ub=np.asarray(distances, dtype=float)[None, :] / n,
        b_ub=np.array([rho]),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    assert result.success
    return float(-result.fun)


def test_zero_radius_is_empirical_probability() -> None:
    distances = np.array([0.0, 0.0, 0.2, 0.8, 1.1])
    assert np.isclose(worst_case_collision_risk(distances, 0.0), 2.0 / 5.0)


def test_ordered_formula_matches_transport_linear_program() -> None:
    rng = np.random.default_rng(21)
    for n in (3, 10, 31):
        for _ in range(30):
            distances = rng.uniform(0.01, 2.0, size=n)
            distances[rng.random(n) < 0.15] = 0.0
            rho = rng.uniform(0.0, 1.2)
            exact = worst_case_collision_risk(distances, rho)
            numerical = risk_by_linear_program(distances, rho)
            assert np.isclose(exact, numerical, atol=1e-9)


def test_critical_radius_is_risk_boundary() -> None:
    distances = np.array([0.0, 0.15, 0.3, 0.8, 1.4, 2.0])
    alpha = 0.4
    critical = critical_wasserstein_radius(distances, alpha)
    assert np.isclose(worst_case_collision_risk(distances, critical), alpha)
    assert worst_case_collision_risk(distances, critical + 1e-4) > alpha
    assert worst_case_collision_risk(distances, max(0.0, critical - 1e-4)) < alpha


def test_margin_and_probability_conditions_agree() -> None:
    distances = np.array([0.0, 0.2, 0.5, 0.9, 1.5])
    critical = critical_wasserstein_radius(distances, 0.4)
    assert dr_safety_margin(distances, 0.4, critical).safe
    assert not dr_safety_margin(distances, 0.4, critical + 1e-3).safe


def test_empirical_violation_cannot_be_repaired_by_dro() -> None:
    distances = np.array([0.0, 0.0, 0.0, 0.4, 0.8])
    result = dr_safety_margin(distances, risk_level=0.4, wasserstein_radius=0.0)
    assert result.empirical_probability == 0.6
    assert not result.safe


def test_worst_case_risk_is_monotone_in_radius() -> None:
    distances = np.array([0.0, 0.1, 0.4, 0.7, 1.2])
    risks = [worst_case_collision_risk(distances, rho) for rho in np.linspace(0.0, 0.8, 30)]
    assert np.all(np.diff(risks) >= -1e-12)
