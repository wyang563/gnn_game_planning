
import jax.numpy as jnp
from typing import List, Dict


def compute_ade(
    ego_traj: jnp.ndarray,
    ego_all_players_ref: jnp.ndarray,
    pos_dim: int = 2
) -> float:
    """
    Compute Average Displacement Error (ADE) for prediction phase.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        ego_all_players_ref: Reference trajectory when all players in mask, shape (T, pos_dim)
        observation_horizon: Number of observation timesteps
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        ADE value
    """
    pred_positions = ego_traj[:, :pos_dim]
    ego_all_players_ref_positions = ego_all_players_ref[:, :pos_dim]
    
    if len(pred_positions) > 0:
        ade = jnp.mean(jnp.linalg.norm(pred_positions - ego_all_players_ref_positions, axis=1))
        return float(ade)
    else:
        return 0.0


def compute_fde(
    ego_traj: jnp.ndarray,
    ego_ref: jnp.ndarray,
    observation_horizon: int,
    pos_dim: int = 2
) -> float:
    """
    Compute Final Displacement Error (FDE) from intended goal.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        ego_ref: Reference trajectory of shape (T, pos_dim)
        observation_horizon: Number of observation timesteps
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        FDE value
    """
    K = observation_horizon
    pred_positions = ego_traj[K:, :pos_dim]
    ego_goal = ego_ref[-1, :pos_dim]
    
    if len(pred_positions) > 0:
        fde = jnp.linalg.norm(pred_positions[-1] - ego_goal)
        return float(fde)
    else:
        return 0.0


def compute_navigation_cost(
    ego_traj: jnp.ndarray,
    ego_ref: jnp.ndarray,
    pos_dim: int = 2
) -> float:
    """
    Compute navigation cost as sum of squared distance to reference.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        ego_ref: Reference trajectory of shape (T, pos_dim)
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        Navigation cost value
    """
    nav_cost = jnp.sum(jnp.linalg.norm(ego_traj[:, :pos_dim] - ego_ref[:, :pos_dim], axis=1) ** 2)
    return float(nav_cost)


def compute_collision_cost(
    ego_traj: jnp.ndarray,
    x_trajs: jnp.ndarray,
    pos_dim: int = 2
) -> float:
    """
    Compute collision cost as sum of exponential proximity costs with all other agents.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        x_trajs: All agent trajectories of shape (N, T, state_dim)
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        Collision cost value
    """
    N, T, _ = x_trajs.shape
    col_cost = 0.0
    
    for t in range(T):
        for j in range(1, N):
            dist_sq = jnp.sum((ego_traj[t, :pos_dim] - x_trajs[j, t, :pos_dim]) ** 2)
            col_cost += jnp.exp(-dist_sq)
    
    return float(col_cost)


def compute_control_cost(ego_control: jnp.ndarray) -> float:
    """
    Compute control cost as sum of squared control magnitudes.
    
    Args:
        ego_control: Control array of shape (T, u_dim)
    
    Returns:
        Control cost value
    """
    ctrl_cost = jnp.sum(jnp.linalg.norm(ego_control, axis=1) ** 2)
    return float(ctrl_cost)


def compute_trajectory_heading(
    ego_traj: jnp.ndarray,
    state_dim: int,
    pos_dim: int = 2
) -> float:
    """
    Compute trajectory heading change (smoothness) as cumulative change in direction.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        state_dim: Dimension of state space
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        Trajectory heading change value
    """
    T = ego_traj.shape[0]
    traj_h = 0.0
    
    for t in range(2, T):
        if state_dim >= 2 * pos_dim:
            # Use velocity if available (velocity starts after position)
            v_curr = ego_traj[t, pos_dim:2*pos_dim]
            v_prev = ego_traj[t-1, pos_dim:2*pos_dim]
            norm_curr = jnp.linalg.norm(v_curr)
            norm_prev = jnp.linalg.norm(v_prev)
            if norm_curr > 1e-6 and norm_prev > 1e-6:
                dir_curr = v_curr / norm_curr
                dir_prev = v_prev / norm_prev
                traj_h += jnp.linalg.norm(dir_curr - dir_prev)
        else:
            # Use position differences
            p_curr = ego_traj[t, :pos_dim] - ego_traj[t-1, :pos_dim]
            p_prev = ego_traj[t-1, :pos_dim] - ego_traj[t-2, :pos_dim]
            norm_curr = jnp.linalg.norm(p_curr)
            norm_prev = jnp.linalg.norm(p_prev)
            if norm_curr > 1e-6 and norm_prev > 1e-6:
                dir_curr = p_curr / norm_curr
                dir_prev = p_prev / norm_prev
                traj_h += jnp.linalg.norm(dir_curr - dir_prev)
    
    return float(traj_h)


def compute_trajectory_length(ego_traj: jnp.ndarray, pos_dim: int = 2) -> float:
    """
    Compute trajectory length as total distance traveled.
    
    Args:
        ego_traj: Ego agent trajectory of shape (T, state_dim)
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        Trajectory length value
    """
    traj_l = jnp.sum(jnp.linalg.norm(jnp.diff(ego_traj[:, :pos_dim], axis=0), axis=1))
    return float(traj_l)


def compute_minimum_distance(
    x_trajs: jnp.ndarray,
    pos_dim: int = 2
) -> float:
    """
    Compute minimum distance to other agents (safety metric, higher is better).
    
    Args:
        x_trajs: All agent trajectories of shape (N, T, state_dim)
        pos_dim: Position dimensionality (2 for 2D, 3 for 3D)
    
    Returns:
        Minimum distance value
    """
    N, T, _ = x_trajs.shape
    
    # Compute pairwise distances for all timesteps at once
    # Shape: (N, 1, T, pos_dim) - (1, N, T, pos_dim) = (N, N, T, pos_dim)
    # After norm: (N, N, T)
    pairwise_distances = jnp.linalg.norm(
        x_trajs[:, None, :, :pos_dim] - x_trajs[None, :, :, :pos_dim], 
        axis=-1
    )
    
    # Mask out self-distances (diagonal) by setting them to infinity
    # Shape: (N, N, 1) to broadcast across time dimension
    id_mask = jnp.eye(N, dtype=bool)[:, :, None]
    pairwise_distances = jnp.where(id_mask, jnp.inf, pairwise_distances)
    
    # Find minimum distance across all agent pairs and all timesteps
    min_dist = float(jnp.min(pairwise_distances))
    
    return min_dist


def compute_consistency(
    simulation_masks: List[jnp.ndarray],
    N: int
) -> float:
    """
    Compute consistency as stability of mask selection over time.
    
    Args:
        simulation_masks: List of masks for each timestep
        N: Number of agents
    
    Returns:
        Consistency value
    """
    if len(simulation_masks) > 1:
        consistency = 0.0
        for t in range(1, len(simulation_masks)):
            mask_diff = jnp.sum(jnp.abs(simulation_masks[t] - simulation_masks[t-1]))
            consistency += 1 - (mask_diff / (N - 1))
        consistency /= (len(simulation_masks) - 1)
        return float(consistency)
    else:
        return 1.0


def compute_avg_num_players_selected(simulation_masks: List[jnp.ndarray]) -> float:
    """
    Compute average number of players selected across all timesteps.
    
    Args:
        simulation_masks: List of masks for each timestep
    
    Returns:
        Average number of players selected
    """
    return float(jnp.mean(jnp.array([jnp.sum(m) for m in simulation_masks])))


def compute_metrics(
    x_trajs: jnp.ndarray,
    control_trajs: jnp.ndarray,
    simulation_masks: List[jnp.ndarray],
    all_players_ref_trajs: jnp.ndarray,
    ref_trajs: jnp.ndarray,
    observation_horizon: int = 10
) -> Dict[str, float]:
    """
    Compute evaluation metrics from the PSN paper.
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim)
        control_trajs: Control array of shape (N, T, u_dim)
        simulation_masks: List of masks for each timestep
        all_players_ref_trajs: Reference trajectories when all players in mask, shape (N, T, pos_dim)
        ref_trajs: Reference trajectories of shape (N, T, pos_dim)
        observation_horizon: Number of observation timesteps
    
    Returns:
        Dictionary of computed metrics
    """
    N, T, state_dim = x_trajs.shape
    
    # Infer position dimensionality from state dimension
    # Assume state = [position, velocity], so pos_dim = state_dim // 2
    pos_dim = state_dim // 2
    
    # For ego agent (agent 0)
    ego_traj = x_trajs[0]  # (T, state_dim)
    ego_control = control_trajs[0]  # (T, u_dim)
    ego_all_players_ref = all_players_ref_trajs[0]  # (T, pos_dim)
    ego_ref = ref_trajs[0]  # (T, pos_dim)
    
    # Compute all metrics using individual functions
    metrics = {
        'ADE': compute_ade(ego_traj, ego_all_players_ref, pos_dim),
        'FDE': compute_fde(ego_traj, ego_ref, observation_horizon, pos_dim),
        'Nav_Cost': compute_navigation_cost(ego_traj, ego_ref, pos_dim),
        'Col_Cost': compute_collision_cost(ego_traj, x_trajs, pos_dim),
        'Ctrl_Cost': compute_control_cost(ego_control),
        'Traj_Heading': compute_trajectory_heading(ego_traj, state_dim, pos_dim),
        'Traj_Length': compute_trajectory_length(ego_traj, pos_dim),
        'Min_Dist': compute_minimum_distance(x_trajs, pos_dim),
        'Consistency': compute_consistency(simulation_masks, N),
        'Avg_Num_Players_Selected': compute_avg_num_players_selected(simulation_masks)
    }
    
    return metrics