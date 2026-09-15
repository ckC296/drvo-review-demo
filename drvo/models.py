"""Minimal planar kinematic models used only by the numerical experiments."""

from dataclasses import dataclass
from typing import Any

import numpy as np


ArrayLike = Any
FloatArray = np.ndarray


@dataclass
class DiscAgent:
    position: FloatArray
    velocity: FloatArray
    radius: float

    @classmethod
    def create(cls, position: ArrayLike, velocity: ArrayLike, radius: float) -> "DiscAgent":
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        if p.shape != (2,) or v.shape != (2,) or radius <= 0.0:
            raise ValueError("invalid disc-agent parameters")
        return cls(p.copy(), v.copy(), float(radius))

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.position += self.velocity * dt


def goal_velocity(position: ArrayLike, goal: ArrayLike, preferred_speed: float) -> FloatArray:
    """Point toward the goal with bounded speed."""
    delta = np.asarray(goal, dtype=float) - np.asarray(position, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return np.zeros(2)
    return min(preferred_speed, distance) * delta / distance
