# loss functions used for MLP and GNN models during training, primarily to calculate similarity loss

import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
from load_config import load_config, get_device_config
from solver.point_agent import PointAgent
from data.ref_traj_data_loading import prepare_batch_for_training
from solver.solve_differentiable import solve_masked_game_differentiable, solve_masked_game_differentiable_parallel
from data.ref_traj_data_loading import extract_observation_trajectory 

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
config = load_config()

# Game parameters
# N_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id
dt = config.game.dt
T_total = config.game.T_total
T_observation = config.psn.observation_length
T_reference = config.game.T_total

# Game solving parameters - get agent-specific config
agent_type = config.game.agent_type
opt_config = getattr(config.optimization, agent_type)
state_dim = opt_config.state_dim
control_dim = opt_config.control_dim
num_iters = opt_config.num_iters
step_size = opt_config.step_size
collision_weight = opt_config.collision_weight
collision_scale = opt_config.collision_scale
ctrl_weight = opt_config.control_weight
device = get_device_config()

Q = jnp.diag(jnp.array(opt_config.Q))  # State cost weights [x, y, vx, vy]
R = jnp.diag(jnp.array(opt_config.R))               # Control cost weights [ax, ay]

def binary_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Binary loss: encourages mask values to be close to 0 or 1."""
    binary_penalty = mask * (1 - mask)
    return jnp.mean(binary_penalty)

def mask_sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Mask sparsity loss: encourages fewer agents to be selected."""
    return jnp.mean(mask)

# ============================================================================
# Similarity loss function helpers
# ============================================================================

def _observations_to_initial_states(obs_row: jnp.ndarray, n_agents: int, obs_input_type: str = "full") -> jnp.ndarray:
    """Convert a flattened observations row into initial 4D states for each agent.
    obs_row: shape (T_observation * N_agents * obs_dim,)
    returns: (N_agents, 4)
    """
    # Determine observation dimension based on input type
    obs_dim = 2 if obs_input_type == "partial" else 4
    
    traj = obs_row.reshape(T_observation, n_agents, obs_dim)
    first = traj[0]  # (N_agents, obs_dim)
    
    if obs_input_type == "partial":
        # For partial observations, we only have position (x, y)
        # Set velocity to zero
        pos = first[:, :2]  # (N_agents, 2)
        vel = jnp.zeros((n_agents, 2))  # (N_agents, 2) - zero velocity
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
                                        ref_ego_traj: jnp.ndarray,
                                        n_agents: int) -> jnp.ndarray:
    """Fully-JAX similarity loss using arrays only (no Python dict access).
    - initial_states: (N_agents, 4)
    - predicted_goals_row: (N_agents * 2,) or (N_agents, 2)
    - predicted_mask_row: (N_agents,)
    - ref_ego_traj: (T_reference, state_dim)
    """
    targets = predicted_goals_row.reshape(n_agents, 2)
    mask_values = predicted_mask_row

    # Use the new differentiable game solver directly

    state_trajectories, _ = solve_masked_game_differentiable_parallel(
        agents,
        [initial_states[i] for i in range(n_agents)],
        targets,
        mask_values,
        num_iters=num_iters,
    )

    # state_trajectories, _ = solve_masked_game_differentiable(
    #     agents,
    #     [initial_states[i] for i in range(n_agents)],
    #     targets,
    #     mask_values,
    #     num_iters=num_iters,
    # )

    ego_traj_masked = state_trajectories[0]
    return similarity_loss(ego_traj_masked, ref_ego_traj)

def batch_similarity_loss(predicted_masks: jnp.ndarray, predicted_goals: jnp.ndarray, batch_data: List[Dict[str, Any]], obs_input_type: str = "full") -> jnp.ndarray:
    """Batch similarity loss: encourages predicted masks to be similar to predicted goals."""
    n_agents = batch_data[0]['metadata']['n_agents']
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
        for _ in range(n_agents)
    ]

    # initialize loss functions
    for agent in shared_agents:
        agent.create_loss_function_mask()

    observations, ref_trajs = prepare_batch_for_training(batch_data, obs_input_type)

    def per_sample(i):
        mask = predicted_masks[i]
        goals = predicted_goals[i]
        obs_row = observations[i]
        ref_ego = ref_trajs[i]
        init_states = _observations_to_initial_states(obs_row, n_agents, obs_input_type)
        return compute_similarity_loss_from_arrays(shared_agents, init_states, goals, mask, ref_ego, n_agents)

    valid_bs = min(predicted_masks.shape[0], observations.shape[0])
    losses = jax.vmap(per_sample)(jnp.arange(valid_bs))
    return jnp.mean(losses)

# ============================================================================
# Ego Cost Loss Functions
# ============================================================================
#
# The ego cost loss functions compute the game-theoretic cost for the ego agent
# when solving the masked game. These functions are used during PSN training to
# evaluate how well the predicted masks enable the ego agent to achieve its goals
# while avoiding collisions with selected agents.
#
# Key differences from similarity loss:
# - Similarity loss: Compares predicted trajectory shape/position with reference
# - Ego cost loss: Evaluates game-theoretic cost (navigation + collision + control)
#
# Functions:
# 1. ego_agent_game_cost: Compute ego agent's total game cost (nav + collision + control)
# 2. prepare_batch_for_ego_cost: Prepare batch data with all agents' reference trajectories
# 3. compute_ego_agent_cost_from_arrays: Solve masked game and compute ego cost (JAX-friendly)
# 4. batch_ego_agent_cost: Batch computation of ego agent cost using vmap
#
# ============================================================================

def ego_agent_game_cost(ego_state_traj: jnp.ndarray, ego_control_traj: jnp.ndarray, 
                       other_agents_trajs: List[jnp.ndarray], mask_values: jnp.ndarray,
                       ref_traj: jnp.ndarray, apply_masks: bool = True) -> jnp.ndarray:
    """
    Compute the ego agent's total game cost.
    
    This function computes the total cost for the ego agent in the masked game,
    which includes navigation cost (tracking reference trajectory), collision 
    avoidance cost (with other agents, weighted by mask values), and control cost.
    
    Args:
        ego_state_traj: Ego agent's state trajectory (T_total, state_dim)
        ego_control_traj: Ego agent's control trajectory (T_total, control_dim)
        other_agents_trajs: List of other agents' state trajectories
        mask_values: Mask values for collision avoidance with other agents
        ref_traj: Reference trajectory for navigation (T_total, state_dim)
        apply_masks: Whether to apply mask values to collision costs (True for game solving, False for training loss)
        
    Returns:
        Total game cost for the ego agent
    """
    T_total_local = ego_state_traj.shape[0]
    
    def single_step_cost(t):
        """Compute cost for a single timestep"""
        ego_state = ego_state_traj[t]  # (state_dim,)
        ego_control = ego_control_traj[t]  # (control_dim,)
        ref_state = ref_traj[t]  # (state_dim,)
        
        # Navigation cost - track reference trajectory
        nav_cost = jnp.sum(jnp.square(ego_state[:2] - ref_state[:2]))
        
        # Collision avoidance cost with other agents (weighted by mask values)
        collision_cost = 0.0
        if len(other_agents_trajs) > 0:
            ego_pos = ego_state[:2]  # (2,)
            
            for i, other_traj in enumerate(other_agents_trajs):
                if i < len(mask_values):
                    other_state = other_traj[t]  # (state_dim,)
                    other_pos = other_state[:2]  # (2,)
                    
                    # Distance squared between ego and other agent
                    distance_squared = jnp.sum(jnp.square(ego_pos - other_pos))
                    
                    # Collision cost - conditionally apply mask based on context
                    if apply_masks:
                        # Game solving: apply mask values
                        mask_value = mask_values[i]
                        collision_cost += (collision_weight * 
                                         mask_value * 
                                         jnp.exp(-collision_scale * distance_squared))
                    else:
                        # Training loss: do not apply mask values
                        collision_cost += (collision_weight * 
                                         jnp.exp(-collision_scale * distance_squared))
        
        # Control cost - penalty on control effort
        ctrl_cost = ctrl_weight * jnp.sum(jnp.square(ego_control))
        
        return nav_cost + collision_cost + ctrl_cost
    
    # Compute cost for all timesteps
    costs = jax.vmap(single_step_cost)(jnp.arange(T_total_local))
    
    # Return total cost (sum over time, scaled by dt)
    return jnp.sum(costs) * dt


def prepare_batch_for_ego_cost(batch_data: List[Dict[str, Any]], obs_input_type: str = "full"):
    """
    Prepare batch data for ego cost computation, extracting all agents' reference trajectories.
    
    Args:
        batch_data: List of reference trajectory samples
        obs_input_type: Observation input type ["full", "partial"]
        
    Returns:
        observations: Batch of observations (batch_size, T_observation * N_agents * obs_dim)
        ref_trajs: Batch of ego reference trajectories (batch_size, T_reference, state_dim)
        all_agents_ref_trajs: List of reference trajectories for all agents (batch_size, List of N_agents trajectories)
    """
    
    # Determine observation dimensions based on input type
    # if obs_input_type == "partial":
    #     obs_dim = 2  # Only position (x, y)
    # else:  # "full"
    #     obs_dim = 4  # Full state (x, y, vx, vy)
    
    batch_obs = []
    batch_ref_traj = []
    batch_all_agents_ref_trajs = []
    n_agents = batch_data[0]['metadata']['n_agents']
    
    for sample_data in batch_data:
        # Use ego_agent_id from config (already imported at top)
        # ego_agent_id is defined globally from config.game.ego_agent_id
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data, obs_input_type)
        batch_obs.append(obs_traj.flatten())
        
        # Extract reference trajectories for all agents
        all_agents_ref_trajs = []
        for agent_id in range(n_agents):
            agent_key = f"agent_{agent_id}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            agent_ref_traj = jnp.array(agent_states)
            
            # Ensure trajectory has correct length
            if agent_ref_traj.shape[0] != T_reference:
                if agent_ref_traj.shape[0] < T_reference:
                    pad_size = T_reference - agent_ref_traj.shape[0]
                    last_state = agent_ref_traj[-1:]
                    padding = jnp.tile(last_state, (pad_size, 1))
                    agent_ref_traj = jnp.concatenate([agent_ref_traj, padding], axis=0)
                else:
                    agent_ref_traj = agent_ref_traj[:T_reference]
            
            all_agents_ref_trajs.append(agent_ref_traj)
        
        # Extract ego reference trajectory
        ego_ref_traj = all_agents_ref_trajs[ego_agent_id]
        batch_ref_traj.append(ego_ref_traj)
        batch_all_agents_ref_trajs.append(all_agents_ref_trajs)
    
    # Convert to JAX arrays
    batch_obs = jnp.stack(batch_obs)
    batch_ref_traj = jnp.stack(batch_ref_traj)
    
    return batch_obs, batch_ref_traj, batch_all_agents_ref_trajs


def compute_ego_agent_cost_from_arrays(agents: list,
                                       initial_states: jnp.ndarray,
                                       predicted_goals_row: jnp.ndarray,
                                       predicted_mask_row: jnp.ndarray,
                                       ref_ego_traj: jnp.ndarray,
                                       ref_other_trajs: jnp.ndarray,
                                       n_agents: int,
                                       apply_masks: bool = True) -> jnp.ndarray:
    """Fully-JAX ego agent cost using arrays only (no Python dict access).
    
    Args:
        agents: List of agent objects (length n_agents)
        initial_states: Initial states for all agents (n_agents, 4)
        predicted_goals_row: Predicted goals for all agents (n_agents * 2,) or (n_agents, 2)
        predicted_mask_row: Predicted mask values for non-ego agents (n_agents - 1,)
        ref_ego_traj: Reference trajectory for ego agent (T_reference, state_dim)
        ref_other_trajs: Reference trajectories for other agents (n_agents-1, T_reference, state_dim)
        n_agents: Number of agents
        apply_masks: Whether to apply mask values to collision costs
        
    Returns:
        Ego agent's total game cost
    """
    targets = predicted_goals_row.reshape(n_agents, 2)
    mask_values = predicted_mask_row

    # Solve the full masked game with all agents to get realistic multi-agent dynamics
    # This ensures the ego agent cost reflects actual game performance, not single-agent optimization
    
    # Convert initial states to list format for game solving
    initial_states_list = [initial_states[i] for i in range(n_agents)]
    
    # Solve the masked game with all agents
    # game_state_trajs, game_control_trajs = solve_masked_game_differentiable(
    #     agents, initial_states_list, targets, mask_values, 
    #     num_iters=num_iters, reference_trajectories=None
    # )

    game_state_trajs, game_control_trajs = solve_masked_game_differentiable_parallel(
        agents,
        initial_states_list,
        targets,
        mask_values,
        num_iters=num_iters,
        reference_trajectories=None
    )
    
    # Extract ego agent's trajectory from the game results
    ego_state_traj = game_state_trajs[0]  # Ego agent is always agent 0
    ego_control_traj = game_control_trajs[0]
    
    # Extract other agents' trajectories from the game results
    other_agents_trajs = game_state_trajs[1:]  # All agents except ego agent
    
    # Compute ego agent's total game cost using actual game results
    return ego_agent_game_cost(ego_state_traj, ego_control_traj, other_agents_trajs, mask_values, ref_ego_traj, apply_masks)

def batch_ego_agent_cost(predicted_masks: jnp.ndarray,
                         predicted_goals: jnp.ndarray,
                         batch_data: List[Dict[str, Any]],
                         obs_input_type: str = "full",
                         apply_masks: bool = True) -> jnp.ndarray:
    """
    Compute ego agent cost for a batch of samples by solving masked games.
    
    Args:
        predicted_masks: Predicted masks from PSN (batch_size, n_agents - 1)
        predicted_goals: Predicted goals from network (batch_size, n_agents * 2)
        batch_data: List of reference trajectory samples
        obs_input_type: Observation input type ["full", "partial"]
        apply_masks: Whether to apply mask values to collision costs (True for game solving, False for training loss)
        
    Returns:
        Average ego agent cost for the batch
    """
    n_agents = batch_data[0]['metadata']['n_agents']
    
    # Build shared agents once (purely static configuration)
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
        for _ in range(n_agents)
    ]

    # initialize loss functions
    for agent in shared_agents:
        agent.create_loss_function_mask()

    # Extract observations and ref trajectories
    observations, ref_trajs, all_agents_ref_trajs = prepare_batch_for_ego_cost(batch_data, obs_input_type)

    # Convert all_agents_ref_trajs to JAX arrays for vmap compatibility
    batch_size = len(all_agents_ref_trajs)
    all_agents_ref_array = jnp.stack([
        jnp.stack([all_agents_ref_trajs[i][j] for j in range(n_agents)]) 
        for i in range(batch_size)
    ])  # Shape: (batch_size, n_agents, T_reference, state_dim)

    def per_sample(i):
        mask = predicted_masks[i]
        goals = predicted_goals[i]
        obs_row = observations[i]
        ref_ego = ref_trajs[i]
        ref_other = all_agents_ref_array[i, 1:, :, :]  # Exclude ego agent (agent 0), shape: (n_agents-1, T_reference, state_dim)
        init_states = _observations_to_initial_states(obs_row, n_agents, obs_input_type)
        return compute_ego_agent_cost_from_arrays(shared_agents, init_states, goals, mask, ref_ego, ref_other, n_agents, apply_masks)

    valid_bs = min(predicted_masks.shape[0], observations.shape[0])
    costs = jax.vmap(per_sample)(jnp.arange(valid_bs))
    return jnp.mean(costs)
