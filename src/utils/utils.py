import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import LQR
import datetime
import os
import random

def random_init(n_agents: int, 
                init_position_range: Tuple[float, float]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    init_ps = []
    goals = []
    
    min_pos, max_pos = init_position_range
    pos_range = max_pos - min_pos
    
    # Minimum distance between agents to avoid initial collisions
    min_distance = 0.5 * pos_range / n_agents  # Scale with number of agents
    
    max_tries = 1000
    
    for _ in range(n_agents):
        # Generate initial position
        init_pos = None
        for _ in range(max_tries):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            
            candidate_pos = jnp.array([x, y])
            
            # Check minimum distance from other agents
            too_close = False
            for existing_pos in init_ps:
                distance = jnp.linalg.norm(candidate_pos - existing_pos)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                init_pos = candidate_pos
                break
        
        init_ps.append(init_pos)
        
        # Generate goal position (different from initial position)
        goal = None
        for _ in range(max_tries):
            goal_x = random.uniform(min_pos, max_pos)
            goal_y = random.uniform(min_pos, max_pos)
            candidate_goal = jnp.array([goal_x, goal_y])
            
            # Ensure goal is far enough from initial position
            distance_to_start = jnp.linalg.norm(candidate_goal - init_pos)
            if distance_to_start > min_distance:
                goal = candidate_goal
                break
        
        goals.append(goal)
    
    return init_ps, goals