import numpy as np
from scipy.optimize import minimize_scalar

from drvo.geometry import FiniteHorizonVO


def numerical_distance_to_union_of_discs(vo: FiniteHorizonVO, w: np.ndarray) -> float:
    """Independent check of Eq. (54): C^T = union_{lambda>=1} B(lambda*c, lambda*r)."""
    axial_scale = max(vo.distance / vo.horizon - vo.cap_radius, 1e-6)
    upper = max(10.0, 4.0 + 4.0 * np.linalg.norm(w) / axial_scale)

    def distance_for_scale(scale: float) -> float:
        return max(np.linalg.norm(w - scale * vo.center) - scale * vo.cap_radius, 0.0)

    result = minimize_scalar(distance_for_scale, bounds=(1.0, upper), method="bounded")
    return min(distance_for_scale(1.0), float(result.fun))


def test_known_membership_cases() -> None:
    vo = FiniteHorizonVO([3.0, 0.0], combined_radius=0.5, horizon=3.0)
    assert vo.contains([1.0, 0.0])
    assert vo.contains([2.0, 0.0])
    assert not vo.contains([0.0, 0.0])
    assert not vo.contains([-1.0, 0.0])
    assert vo.distance_to_set([1.0, 0.0]) == 0.0
    assert np.isclose(vo.distance_to_set([0.0, 0.0]), (3.0 - 0.5) / 3.0)


def test_tangent_points_and_rays_are_on_boundary() -> None:
    vo = FiniteHorizonVO([2.4, -0.7], combined_radius=0.6, horizon=2.5)
    for point in (vo.q_plus, vo.q_minus, 3.0 * vo.q_plus, 2.0 * vo.q_minus):
        assert vo.contains(point)
        assert vo.distance_to_set(point) == 0.0


def test_closed_form_matches_independent_numerical_projection() -> None:
    rng = np.random.default_rng(9)
    for _ in range(12):
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.2, 0.8)
        distance = rng.uniform(radius + 0.3, 5.0)
        p = distance * np.array([np.cos(angle), np.sin(angle)])
        vo = FiniteHorizonVO(p, radius, rng.uniform(0.5, 4.0))
        for w in rng.uniform(-3.0, 3.0, size=(20, 2)):
            closed_form = vo.distance_to_set(w)
            numerical = numerical_distance_to_union_of_discs(vo, w)
            assert np.isclose(closed_form, numerical, atol=2e-5, rtol=2e-5)


def test_translation_identity_for_obstacle_velocity_samples() -> None:
    vo = FiniteHorizonVO([3.0, 1.0], 0.5, 2.0)
    v = np.array([0.8, 0.1])
    samples = np.array([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4]])
    batch = vo.sample_distances(v, samples)
    direct = np.array([vo.distance_to_set(v - u) for u in samples])
    assert np.allclose(batch, direct)
