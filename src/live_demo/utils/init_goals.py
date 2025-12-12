import random
from typing import Tuple, List, Union
import math

def sim_random_init(
    n_agents: int,
    xy_position_range: Tuple[float, float],
    z_position_range: Tuple[float, float],
    min_distance: float = 0.5,
    generate_init_positions: bool = False,
) -> Union[List[List[float]], Tuple[List[List[float]], List[List[float]]]]:
    """
    Generate random 3D goal positions for agents with a minimum distance constraint.
    Optionally also generate random initial positions.

    Args:
        n_agents: Number of agents.
        xy_position_range: (min_xy, max_xy) bounds for x and y coordinates.
        z_position_range: (min_z, max_z) bounds for z coordinate.
        min_distance: Minimum allowed distance between any two goals in the xy plane.
                      This ensures drones don't land on top of each other.
        generate_init_positions: If True, also generate random initial positions.

    Returns:
        If generate_init_positions is False:
            List of 3D goal positions, where each goal is [x, y, z].
        If generate_init_positions is True:
            Tuple of (init_positions, goals), where both are lists of 3D positions.
    """
    
    def generate_positions(n: int, existing_positions: List[List[float]] = None) -> List[List[float]]:
        """Helper function to generate n random positions with minimum distance constraint."""
        positions: List[List[float]] = []
        
        min_xy, max_xy = xy_position_range
        min_z, max_z = z_position_range
        
        max_tries = 1000
        
        for _ in range(n):
            position = None
            for _ in range(max_tries):
                candidate_position = [
                    random.uniform(min_xy, max_xy),  # x
                    random.uniform(min_xy, max_xy),  # y
                    random.uniform(min_z, max_z),    # z
                ]
                
                # Check minimum distance in xy plane from other positions
                too_close = False
                for existing_position in positions:
                    # Only check xy plane distance (ignore z coordinate)
                    xy_distance = math.sqrt(
                        (candidate_position[0] - existing_position[0]) ** 2 +
                        (candidate_position[1] - existing_position[1]) ** 2
                    )
                    if xy_distance < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    position = candidate_position
                    break
            
            # If we couldn't find a valid position after max_tries, use the last candidate
            if position is None:
                position = candidate_position
            
            positions.append(position)
        
        return positions
    
    # Generate goals
    goals = generate_positions(n_agents)
    
    # Generate initial positions if requested
    if generate_init_positions:
        init_positions = generate_positions(n_agents)
        return init_positions, goals
    
    return goals