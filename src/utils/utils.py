import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import LQR
import datetime
import os
import random
import yaml
import argparse
from policies import *

def origin_init_collision(n_agents: int, 
                init_position_range: Tuple[float, float]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    init_ps = []
    goals = []
    
    min_pos, max_pos = init_position_range
    pos_range = max_pos - min_pos
    
    # Minimum distance between agents to avoid initial collisions
    min_distance = 0.5 * pos_range / n_agents  # Scale with number of agents
    
    max_tries = 1000
    
    # 1) Generate all initial positions with spacing
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

    # 2) For each agent, choose goal along the same line through origin: goal = -c * init
    #    Pick c > 0 within bounds so that goal stays inside [min_pos, max_pos]^2
    for i in range(n_agents):
        x0, y0 = float(init_ps[i][0]), float(init_ps[i][1])

        # Determine feasible interval for c such that goal = -c * init lies within bounds
        c_lower = 0.0
        c_upper = float('inf')

        # x-dimension constraint
        if x0 > 0:
            # -c*x0 in [min_pos, max_pos] => c in [-max_pos/x0, -min_pos/x0]
            c_lower = max(c_lower, -max_pos / x0)
            c_upper = min(c_upper, -min_pos / x0)
        elif x0 < 0:
            # -c*x0 in [min_pos, max_pos] => c in [-min_pos/x0, -max_pos/x0]
            c_lower = max(c_lower, -min_pos / x0)
            c_upper = min(c_upper, -max_pos / x0)
        # if x0 == 0, no constraint from x

        # y-dimension constraint
        if y0 > 0:
            c_lower = max(c_lower, -max_pos / y0)
            c_upper = min(c_upper, -min_pos / y0)
        elif y0 < 0:
            c_lower = max(c_lower, -min_pos / y0)
            c_upper = min(c_upper, -max_pos / y0)
        # if y0 == 0, no constraint from y

        # Ensure positive and valid interval
        c_lower = max(c_lower, 0.0)
        if not (c_lower < c_upper and c_upper > 0):
            # Fallback: if constraints degenerate, pick c = 1 and clip goal into bounds
            c = 1.0
            goal = -c * jnp.array([x0, y0])
            goal = jnp.clip(goal, min_pos, max_pos)
            goals.append(goal)
            continue

        # Prefer c around 1 with slight randomness, then clamp into [c_lower, c_upper]
        desired_c = 1.0 + random.uniform(-0.2, 0.2)
        # Keep a tiny margin inside bounds to avoid floating point boundary hits
        margin = 1e-6
        c = min(max(desired_c, c_lower + margin), c_upper - margin)

        goal = -c * jnp.array([x0, y0])
        goals.append(goal)
    
    return init_ps, goals

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def random_init(n_agents: int, 
                init_position_range: Tuple[float, float]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Generate random initial positions and goals for agents with minimum distance constraint.
    
    Args:
        n_agents: Number of agents
        init_position_range: Tuple of (min_pos, max_pos) for position bounds
        
    Returns:
        Tuple of (initial_positions, goals) where each is a list of jnp.ndarray
    """
    init_ps = []
    goals = []
    
    min_pos, max_pos = init_position_range
    pos_range = max_pos - min_pos
    
    # Minimum distance between agents (0.05 * init_range_x)
    min_distance = 0.03 * pos_range
    
    max_tries = 1000
    
    # Generate initial positions with minimum distance constraint
    for _ in range(n_agents):
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
        
        # If we couldn't find a valid position after max_tries, use the last candidate
        if init_pos is None:
            init_pos = candidate_pos
        
        init_ps.append(init_pos)
    
    # Generate random goals with minimum distance constraint
    for _ in range(n_agents):
        goal = None
        for _ in range(max_tries):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            
            candidate_goal = jnp.array([x, y])
            
            # Check minimum distance from other goals
            too_close = False
            for existing_goal in goals:
                distance = jnp.linalg.norm(candidate_goal - existing_goal)
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
    
    return init_ps, goals

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments containing config file path
    """
    parser = argparse.ArgumentParser(description='Run LQR-based multi-agent simulation')
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to YAML configuration file (default: configs/point_agent_small.yaml)'
    )
    return parser.parse_args()

def get_masking_function(masking_func_str: str):
    pass