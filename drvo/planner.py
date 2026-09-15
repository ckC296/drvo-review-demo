"""Discrete candidate-velocity implementation of manuscript Algorithm 1."""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import FiniteHorizonVO
from .risk import RiskResult, dr_safety_margin


ArrayLike = Any
FloatArray = np.ndarray


@dataclass(frozen=True)
class ObstacleConstraint:
    """Geometry, samples and risk parameters for one obstacle."""

    geometry: FiniteHorizonVO
    velocity_samples: ArrayLike
    risk_level: float
    wasserstein_radius: float


@dataclass(frozen=True)
class PlanResult:
    """Selected command and auditable results for all obstacle checks."""

    velocity: Optional[FloatArray]
    feasible: bool
    cost: float
    checked_candidates: int
    obstacle_risks: Tuple[RiskResult, ...]


def generate_polar_candidates(
    max_speed: float,
    speed_levels: int = 8,
    angle_samples: int = 72,
    desired_velocity: Optional[ArrayLike] = None,
) -> FloatArray:
    """Generate a polar velocity set and explicitly include zero/desired velocity."""
    if max_speed <= 0.0 or speed_levels < 1 or angle_samples < 3:
        raise ValueError("invalid polar candidate-set parameters")
    candidates = [np.zeros(2)]
    for speed in np.linspace(max_speed / speed_levels, max_speed, speed_levels):
        for angle in np.linspace(-np.pi, np.pi, angle_samples, endpoint=False):
            candidates.append(speed * np.array([np.cos(angle), np.sin(angle)]))
    if desired_velocity is not None:
        desired = np.asarray(desired_velocity, dtype=float)
        if desired.shape != (2,):
            raise ValueError("desired_velocity must be two-dimensional")
        norm = float(np.linalg.norm(desired))
        if norm <= max_speed + 1e-12:
            candidates.append(desired.copy())
    return np.unique(np.round(np.asarray(candidates), decimals=12), axis=0)


def dynamically_admissible(
    candidates: ArrayLike,
    previous_velocity: ArrayLike,
    max_acceleration: Optional[float],
    dt: float,
) -> FloatArray:
    """Apply the acceleration part of manuscript Eq. (32)."""
    values = np.asarray(candidates, dtype=float)
    previous = np.asarray(previous_velocity, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or previous.shape != (2,):
        raise ValueError("candidate velocities must have shape (K, 2)")
    if max_acceleration is None:
        return values
    if max_acceleration <= 0.0 or dt <= 0.0:
        raise ValueError("max_acceleration and dt must be positive")
    mask = np.linalg.norm(values - previous, axis=1) <= max_acceleration * dt + 1e-12
    return values[mask]


def select_velocity(
    candidates: ArrayLike,
    desired_velocity: ArrayLike,
    previous_velocity: ArrayLike,
    obstacles: Sequence[ObstacleConstraint],
    smoothing_weight: float = 0.05,
    max_acceleration: Optional[float] = None,
    dt: float = 0.1,
) -> PlanResult:
    """Return the first safe velocity after sorting candidates by Eq. (107)."""
    desired = np.asarray(desired_velocity, dtype=float)
    previous = np.asarray(previous_velocity, dtype=float)
    if desired.shape != (2,) or previous.shape != (2,):
        raise ValueError("desired_velocity and previous_velocity must be two-dimensional")
    if smoothing_weight < 0.0:
        raise ValueError("smoothing_weight must be nonnegative")
    admissible = dynamically_admissible(candidates, previous, max_acceleration, dt)
    if admissible.size == 0:
        return PlanResult(None, False, float("inf"), 0, tuple())

    costs = np.sum((admissible - desired) ** 2, axis=1)
    costs += smoothing_weight * np.sum((admissible - previous) ** 2, axis=1)
    order = np.argsort(costs, kind="stable")

    checked = 0
    for index in order:
        candidate = admissible[index]
        checked += 1
        diagnostics: List[RiskResult] = []
        safe = True
        for obstacle in obstacles:
            distances = obstacle.geometry.sample_distances(candidate, obstacle.velocity_samples)
            result = dr_safety_margin(
                distances,
                obstacle.risk_level,
                obstacle.wasserstein_radius,
            )
            diagnostics.append(result)
            if not result.safe:
                safe = False
                break
        if safe:
            return PlanResult(candidate.copy(), True, float(costs[index]), checked, tuple(diagnostics))
    return PlanResult(None, False, float("inf"), checked, tuple())
