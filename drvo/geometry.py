"""Closed-form geometry for the finite-horizon velocity obstacle.

The notation follows Sec. II-D and II-E of ``main.tex``.  The class stores
the geometry that depends only on relative position, combined radius and
collision horizon, so it can be reused for all velocity samples in one
control cycle.
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


ArrayLike = Any
FloatArray = np.ndarray


@dataclass(frozen=True)
class FiniteHorizonVO:
    """Finite-horizon collision-velocity set for two discs.

    Parameters
    ----------
    relative_position:
        Obstacle position minus robot position, :math:`p_j-p_A`.
    combined_radius:
        Robot radius + obstacle radius + optional safety margin.
    horizon:
        Constant-velocity collision-checking horizon ``T``.
    """

    relative_position: ArrayLike
    combined_radius: float
    horizon: float
    tolerance: float = 1e-10

    p: FloatArray = field(init=False, repr=False)
    distance: float = field(init=False)
    center: FloatArray = field(init=False)
    cap_radius: float = field(init=False)
    e: FloatArray = field(init=False, repr=False)
    n: FloatArray = field(init=False, repr=False)
    sin_theta: float = field(init=False)
    cos_theta: float = field(init=False)
    tan_theta: float = field(init=False)
    ell: float = field(init=False)
    u_plus: FloatArray = field(init=False, repr=False)
    u_minus: FloatArray = field(init=False, repr=False)
    q_plus: FloatArray = field(init=False, repr=False)
    q_minus: FloatArray = field(init=False, repr=False)
    x_tangent: float = field(init=False)

    def __post_init__(self) -> None:
        p = np.asarray(self.relative_position, dtype=float)
        if p.shape != (2,):
            raise ValueError("relative_position must be a two-dimensional vector")
        radius = float(self.combined_radius)
        horizon = float(self.horizon)
        distance = float(np.linalg.norm(p))
        if radius <= 0.0:
            raise ValueError("combined_radius must be positive")
        if horizon <= 0.0:
            raise ValueError("horizon must be positive")
        if distance <= radius:
            raise ValueError("the robot and obstacle must initially be collision-free (D > R)")

        e = p / distance
        n = np.array([-e[1], e[0]])
        sin_theta = radius / distance
        cos_theta = np.sqrt(distance * distance - radius * radius) / distance
        ell = np.sqrt(distance * distance - radius * radius) / horizon
        u_plus = cos_theta * e + sin_theta * n
        u_minus = cos_theta * e - sin_theta * n

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "center", p / horizon)
        object.__setattr__(self, "cap_radius", radius / horizon)
        object.__setattr__(self, "e", e)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "sin_theta", sin_theta)
        object.__setattr__(self, "cos_theta", cos_theta)
        object.__setattr__(self, "tan_theta", radius / np.sqrt(distance * distance - radius * radius))
        object.__setattr__(self, "ell", ell)
        object.__setattr__(self, "u_plus", u_plus)
        object.__setattr__(self, "u_minus", u_minus)
        object.__setattr__(self, "q_plus", ell * u_plus)
        object.__setattr__(self, "q_minus", ell * u_minus)
        object.__setattr__(self, "x_tangent", (distance * distance - radius * radius) / (distance * horizon))

    def components(self, relative_velocity: ArrayLike) -> Tuple[float, float]:
        """Return longitudinal and transverse coordinates ``(x, y)``."""
        w = self._vector(relative_velocity)
        return float(self.e @ w), float(self.n @ w)

    def contains(self, relative_velocity: ArrayLike) -> bool:
        """Evaluate the exact membership test in manuscript Eq. (70)."""
        w = self._vector(relative_velocity)
        x, y = self.components(w)
        tol = self.tolerance
        in_cap = (
            np.linalg.norm(w - self.center) <= self.cap_radius + tol
            and x <= self.x_tangent + tol
        )
        in_cone = (
            abs(y) <= x * self.tan_theta + tol
            and x >= self.x_tangent - tol
        )
        return bool(in_cap or in_cone)

    def boundary_distances(self, relative_velocity: ArrayLike) -> Tuple[float, float, float]:
        """Return distances to the circular cap and the two tangent rays."""
        w = self._vector(relative_velocity)

        s_plus = max(self.ell, float(self.u_plus @ w))
        s_minus = max(self.ell, float(self.u_minus @ w))
        d_plus = float(np.linalg.norm(w - s_plus * self.u_plus))
        d_minus = float(np.linalg.norm(w - s_minus * self.u_minus))

        z = w - self.center
        z_norm = float(np.linalg.norm(z))
        radial_on_cap = (
            z_norm > self.tolerance
            and float(self.e @ z) / z_norm <= -self.sin_theta + self.tolerance
        )
        if radial_on_cap:
            d_cap = abs(z_norm - self.cap_radius)
        else:
            d_cap = min(
                float(np.linalg.norm(w - self.q_plus)),
                float(np.linalg.norm(w - self.q_minus)),
            )
        return d_cap, d_plus, d_minus

    def distance_to_set(self, relative_velocity: ArrayLike) -> float:
        """Closed-form Euclidean distance in manuscript Proposition 1."""
        w = self._vector(relative_velocity)
        if self.contains(w):
            return 0.0
        return min(self.boundary_distances(w))

    def sample_distances(self, robot_velocity: ArrayLike, obstacle_velocity_samples: ArrayLike) -> FloatArray:
        """Compute all ``d_j,i(v) = dist(v-u_hat_j,i, C_j^T)``."""
        v = self._vector(robot_velocity)
        samples = np.asarray(obstacle_velocity_samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("obstacle_velocity_samples must have shape (N, 2)")
        w = v[None, :] - samples
        x = w @ self.e
        y = w @ self.n
        cap_norm = np.linalg.norm(w - self.center, axis=1)
        inside = (
            (cap_norm <= self.cap_radius + self.tolerance)
            & (x <= self.x_tangent + self.tolerance)
        ) | (
            (np.abs(y) <= x * self.tan_theta + self.tolerance)
            & (x >= self.x_tangent - self.tolerance)
        )

        s_plus = np.maximum(self.ell, w @ self.u_plus)
        s_minus = np.maximum(self.ell, w @ self.u_minus)
        d_plus = np.linalg.norm(w - s_plus[:, None] * self.u_plus, axis=1)
        d_minus = np.linalg.norm(w - s_minus[:, None] * self.u_minus, axis=1)

        z = w - self.center
        z_norm = np.linalg.norm(z, axis=1)
        radial_on_cap = np.zeros(len(samples), dtype=bool)
        nonzero = z_norm > self.tolerance
        radial_on_cap[nonzero] = (
            (z[nonzero] @ self.e) / z_norm[nonzero]
            <= -self.sin_theta + self.tolerance
        )
        d_cap = np.minimum(
            np.linalg.norm(w - self.q_plus, axis=1),
            np.linalg.norm(w - self.q_minus, axis=1),
        )
        d_cap[radial_on_cap] = np.abs(z_norm[radial_on_cap] - self.cap_radius)
        distances = np.minimum(d_cap, np.minimum(d_plus, d_minus))
        distances[inside] = 0.0
        return distances

    @staticmethod
    def _vector(value: ArrayLike) -> FloatArray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (2,):
            raise ValueError("velocity must be a two-dimensional vector")
        return vector
