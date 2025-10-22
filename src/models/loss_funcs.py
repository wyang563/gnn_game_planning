# loss functions used for MLP and GNN models during training, primarily to calculate similarity loss

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
from load_config import load_config, get_device_config
from solver.point_agent import PointAgent
from data.ref_traj_data_loading import prepare_batch_for_training
from solver.solve_differentiable import solve_masked_game_differentiable

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
config = load_config()

# Game parameters
N_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id
dt = config.game.dt
T_total = config.game.T_total
T_observation = config.psn.observation_length
T_reference = config.game.T_total
state_dim = config.game.state_dim
control_dim = config.game.control_dim

# Game solving parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size
collision_weight = config.optimization.collision_weight
collision_scale = config.optimization.collision_scale
ctrl_weight = config.optimization.control_weight
device = get_device_config()

Q = jnp.diag(jnp.array(config.optimization.Q))  # State cost weights [x, y, vx, vy]
R = jnp.diag(jnp.array(config.optimization.R))               # Control cost weights [ax, ay]

def binary_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Binary loss: encourages mask values to be close to 0 or 1."""
    binary_penalty = mask * (1 - mask)
    return jnp.mean(binary_penalty)

def mask_sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Mask sparsity loss: encourages fewer agents to be selected."""
    return jnp.mean(mask)

# Similarity loss function helpers

def _observations_to_initial_states(obs_row: jnp.ndarray, obs_input_type: str = "full") -> jnp.ndarray:
    """Convert a flattened observations row into initial 4D states for each agent.
    obs_row: shape (T_observation * N_agents * obs_dim,)
    returns: (N_agents, 4)
    """
    # Determine observation dimension based on input type
    obs_dim = 2 if obs_input_type == "partial" else 4
    
    traj = obs_row.reshape(T_observation, N_agents, obs_dim)
    first = traj[0]  # (N_agents, obs_dim)
    
    if obs_input_type == "partial":
        # For partial observations, we only have position (x, y)
        # Set velocity to zero
        pos = first[:, :2]  # (N_agents, 2)
        vel = jnp.zeros((N_agents, 2))  # (N_agents, 2) - zero velocity
        return jnp.concatenate([pos, vel], axis=-1)
    else:
        # For full observations, we have position and velocity
        pos = first[:, :2]
        vel = first[:, 2:4]
        return jnp.concatenate([pos, vel], axis=-1)

def similarity_loss(pred_traj: jnp.ndarray, target_traj: jnp.ndarray) -> jnp.ndarray:
    """
    Similarity loss: compare predicted trajectory with target trajectory.
    
    This loss compares the predicted trajectory from the masked game solver
    with the reference trajectory to measure how well the PSN's mask selection
    enables the game solver to reproduce the desired behavior.
    
    Args:
        pred_traj: Predicted trajectory from masked game (T_total, state_dim)
        target_traj: Reference trajectory (T_reference, state_dim)
        
    Returns:
        Similarity loss value comparing predicted vs reference trajectory
    """
    # Extract positions (first 2 dimensions) for comparison
    pred_positions = pred_traj[:, :2]  # (T_total, 2)
    target_positions = target_traj[:, :2]  # (T_reference, 2)
    
    # Ensure both trajectories have the same length for comparison
    min_length = min(pred_positions.shape[0], target_positions.shape[0])
    
    if min_length == 0:
        return jnp.array(100.0)  # High loss if no valid comparison possible
    
    # Use only the first min_length steps for comparison
    pred_positions_matched = pred_positions[:min_length]  # (min_length, 2)
    target_positions_matched = target_positions[:min_length]  # (min_length, 2)
    
    # Compute position-wise distance
    position_diff = pred_positions_matched - target_positions_matched
    distances = jnp.linalg.norm(position_diff, axis=-1)  # (min_length,)
    
    # Return mean distance
    return jnp.mean(distances)

def compute_similarity_loss_from_arrays(agents: list,
                                        initial_states: jnp.ndarray,
                                        predicted_goals_row: jnp.ndarray,
                                        predicted_mask_row: jnp.ndarray,
                                        ref_ego_traj: jnp.ndarray) -> jnp.ndarray:
    """Fully-JAX similarity loss using arrays only (no Python dict access).
    - initial_states: (N_agents, 4)
    - predicted_goals_row: (N_agents * 2,) or (N_agents, 2)
    - predicted_mask_row: (N_agents - 1,)
    - ref_ego_traj: (T_reference, state_dim)
    """
    targets = predicted_goals_row.reshape(N_agents, 2)
    mask_values = predicted_mask_row

    # Use the new differentiable game solver directly
    state_trajectories, _ = solve_masked_game_differentiable(
        agents,
        [initial_states[i] for i in range(N_agents)],
        targets,
        mask_values,
        num_iters=num_iters,
    )

    ego_traj_masked = state_trajectories[0]
    return similarity_loss(ego_traj_masked, ref_ego_traj)

def batch_similarity_loss(predicted_masks: jnp.ndarray, predicted_goals: jnp.ndarray, batch_data: List[Dict[str, Any]], obs_input_type: str = "full") -> jnp.ndarray:
    """Batch similarity loss: encourages predicted masks to be similar to predicted goals."""
    shared_agents = [
        PointAgent(
            dt=dt, 
            x_dim=4, 
            u_dim=2, 
            Q=Q, 
            R=R, 
            collision_weight=collision_weight, 
            collision_scale=collision_scale, 
            ctrl_weight=ctrl_weight, 
            device=device
        ) 
        for _ in range(N_agents)
    ]
    observations, ref_trajs = prepare_batch_for_training(batch_data, obs_input_type)

    def per_sample(i):
        mask = predicted_masks[i]
        goals = predicted_goals[i]
        obs_row = observations[i]
        ref_ego = ref_trajs[i]
        init_states = _observations_to_initial_states(obs_row, obs_input_type)
        return compute_similarity_loss_from_arrays(shared_agents, init_states, goals, mask, ref_ego)

    valid_bs = min(predicted_masks.shape[0], observations.shape[0])
    losses = jax.vmap(per_sample)(jnp.arange(valid_bs))
    return jnp.mean(losses)


