import random
from typing import Tuple, List
import math

def random_init(
    n_agents: int,
    xy_position_range: Tuple[float, float],
    z_position_range: Tuple[float, float],
    min_distance: float = 0.5,
) -> List[List[float]]:
    """
    Generate random 3D goal positions for agents with a minimum distance constraint.

    Args:
        n_agents: Number of agents.
        xy_position_range: (min_xy, max_xy) bounds for x and y coordinates.
        z_position_range: (min_z, max_z) bounds for z coordinate.
        min_distance: Minimum allowed distance between any two goals.

    Returns:
        List of 3D goal positions, where each goal is [x, y, z].
    """
    goals: List[List[float]] = []

    min_xy, max_xy = xy_position_range
    min_z, max_z = z_position_range

    max_tries = 1000

    # Generate random 3D goals with minimum distance constraint
    for _ in range(n_agents):
        goal = None
        for _ in range(max_tries):
            candidate_goal = [
                random.uniform(min_xy, max_xy),  # x
                random.uniform(min_xy, max_xy),  # y
                random.uniform(min_z, max_z),    # z
            ]

            # Check minimum distance from other goals
            too_close = False
            for existing_goal in goals:
                distance = math.sqrt(
                    sum((c1 - c2) ** 2 for c1, c2 in zip(candidate_goal, existing_goal))
                )
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                goal = candidate_goal
                break

        # If we couldn't find a valid goal after max_tries, use the last candidate
        if goal is None:
            goal = candidate_goal

        goals.append(goal)

    return goals