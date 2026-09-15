"""Small, hardware-independent DR-VO demonstration."""

import numpy as np

from drvo.geometry import FiniteHorizonVO
from drvo.risk import dr_safety_margin, worst_case_collision_risk


def main() -> None:
    # Relative obstacle position, combined radius, and prediction horizon.
    vo = FiniteHorizonVO(relative_position=[3.0, 0.0], combined_radius=0.5, horizon=3.0)
    candidate_velocity = np.array([0.8, 0.0])
    predicted_obstacle_velocities = np.array(
        [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.05], [0.0, -0.1]]
    )

    distances = vo.sample_distances(candidate_velocity, predicted_obstacle_velocities)
    risk = worst_case_collision_risk(distances, wasserstein_radius=0.02)
    margin = dr_safety_margin(
        distances, risk_level=0.1, wasserstein_radius=0.02
    )

    print("candidate velocity:", candidate_velocity)
    print("sample distances:", np.round(distances, 4))
    print(f"worst-case collision risk: {risk:.4f}")
    print("safe:", margin.safe)


if __name__ == "__main__":
    main()
