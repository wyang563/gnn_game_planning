import os
import glob
import json
from typing import Tuple, List, Dict, Any
import jax.numpy as jnp
from load_config import load_config

# ============================================================================
# mostly just data loading and preparation function helpers for training MLP and GNN model
# ============================================================================

# Load configuration
config = load_config()

# Constants from config
N_agents = config.game.N_agents
T_observation = config.psn.observation_length
T_reference = config.game.T_total
batch_size = config.psn.batch_size
state_dim = config.game.state_dim

def load_reference_trajectories(data_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load reference trajectories from directory containing individual JSON files and split into training and validation sets."""
    
    # Find all ref_traj_sample_*.json files in the directory
    pattern = os.path.join(data_dir, "ref_traj_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No ref_traj_sample_*.json files found in directory: {data_dir}")
    
    # Load all samples
    reference_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
                reference_data.append(sample_data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Loaded {len(reference_data)} reference trajectory samples from {data_dir}")
    print(f"Sample files: {len(json_files)} found, {len(reference_data)} loaded successfully")
    
    # Split data: Load from config
    total_samples = len(reference_data)
    train_size = int(total_samples * config.training.train_samples)
    val_size = total_samples - train_size

    reference_data.sort(key=lambda x: x.get('sample_id', 0))
    
    training_data = reference_data[:train_size]
    validation_data = reference_data[train_size:train_size + val_size]
    
    print(f"Training samples: {len(training_data)} (first {len(training_data)} samples)")
    print(f"Validation samples: {len(validation_data)} (samples {len(training_data)} to {len(training_data) + len(validation_data) - 1})")
    
    return training_data, validation_data

def extract_observation_trajectory(sample_data: Dict[str, Any], obs_input_type: str = "full") -> jnp.ndarray:
    """
    Extract observation trajectory (first 10 steps) for all agents.
    
    Args:
        sample_data: Reference trajectory sample
        obs_input_type: Observation input type ["full", "partial"]
        
    Returns:
        observation_trajectory: Observation trajectory 
            - If obs_input_type="full": (T_observation, N_agents, 4)
            - If obs_input_type="partial": (T_observation, N_agents, 2)
    """
    # Determine output dimensions based on observation type
    if obs_input_type == "partial":
        output_dim = 2  # Only position (x, y)
    else:  # "full"
        output_dim = 4  # Full state (x, y, vx, vy)
    
    # Initialize array to store all agent states
    # Shape: (T_observation, N_agents, output_dim)
    observation_trajectory = jnp.zeros((T_observation, N_agents, output_dim))
    
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        states = sample_data["trajectories"][agent_key]["states"]
        # Take first T_observation steps
        agent_states = jnp.array(states[:T_observation])  # (T_observation, state_dim)
        
        # Extract relevant dimensions based on observation type
        if obs_input_type == "partial":
            # Only use position (x, y) - first 2 dimensions
            agent_obs = agent_states[:, :2]  # (T_observation, 2)
        else:  # "full"
            # Use full state (x, y, vx, vy) - all 4 dimensions
            agent_obs = agent_states[:, :4]  # (T_observation, 4)
        
        # Place in the correct position: (T_observation, N_agents, output_dim)
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_obs)
    
    return observation_trajectory

def extract_ego_reference_trajectory(sample_data: Dict[str, Any], ego_agent_id: int) -> jnp.ndarray:
    """
    Extract reference trajectory for the ego agent.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        
    Returns:
        Reference trajectory for ego agent (T_reference, state_dim)
    """
    ego_key = f"agent_{ego_agent_id}"
    ego_states = sample_data["trajectories"][ego_key]["states"]
    return jnp.array(ego_states)

def prepare_batch_for_training(batch_data: List[Dict[str, Any]], obs_input_type: str = "full") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Determine observation dimensions based on input type
    if obs_input_type == "partial":
        obs_dim = 2  # Only position (x, y)
    else:  # "full"
        obs_dim = 4  # Full state (x, y, vx, vy)

    batch_obs = []
    batch_ref_traj = []   

    for sample_data in batch_data:
        ego_agent_id = config.game.ego_agent_id
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data, obs_input_type)
        batch_obs.append(obs_traj.flatten())
        
        # Extract ego reference trajectory
        ego_ref_traj = extract_ego_reference_trajectory(sample_data, ego_agent_id)
        if ego_ref_traj.shape[0] != T_reference:
            if ego_ref_traj.shape[0] < T_reference:
                pad_size = T_reference - ego_ref_traj.shape[0]
                last_state = ego_ref_traj[-1:]
                padding = jnp.tile(last_state, (pad_size, 1))
                ego_ref_traj = jnp.concatenate([ego_ref_traj, padding], axis=0)
            else:
                ego_ref_traj = ego_ref_traj[:T_reference]
        batch_ref_traj.append(ego_ref_traj)
    
    # Pad batch if necessary
    if len(batch_obs) < batch_size:
        pad_size = batch_size - len(batch_obs)
        obs_pad = jnp.zeros((pad_size, T_observation * N_agents * obs_dim))
        batch_obs.extend([obs_pad[i] for i in range(pad_size)])
        ref_pad = jnp.zeros((pad_size, T_reference, state_dim))
        batch_ref_traj.extend([ref_pad[i] for i in range(pad_size)])
    
    # Convert to JAX arrays
    batch_obs = jnp.stack(batch_obs)
    batch_ref_traj = jnp.stack(batch_ref_traj)
    
    return batch_obs, batch_ref_traj

def extract_true_goals_from_batch(batch_data: List[Dict[str, Any]]) -> jnp.ndarray:
    """
    Extract true goals from batch data for training with true goals.
    
    Args:
        batch_data: List of sample data dictionaries
        
    Returns:
        Array of true goals (batch_size, N_agents * 2)
    """
    batch_goals = []
    
    for sample_data in batch_data:
        # Extract goals for all agents from the reference trajectory
        sample_goals = []
        for agent_idx in range(N_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            # Use the final state position as the goal
            if len(agent_states) > 0:
                final_state = agent_states[-1]
                goal_pos = jnp.array([final_state[0], final_state[1]])  # [x, y] from [x, y, vx, vy]
            else:
                # Fallback to zero if no states available
                goal_pos = jnp.array([0.0, 0.0])
            
            sample_goals.append(goal_pos)
        
        # Flatten all agent goals into a single array
        sample_goals_flat = jnp.concatenate(sample_goals)  # (N_agents * 2,)
        batch_goals.append(sample_goals_flat)
    
    # Stack all samples into a batch
    batch_goals_array = jnp.stack(batch_goals)  # (batch_size, N_agents * 2)
    return batch_goals_array
