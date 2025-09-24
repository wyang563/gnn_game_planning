import jax.numpy as jnp
from jax import jit, vmap, grad, lax


def nearest_neighbors(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    """
    Find the top_k nearest neighbors for each agent based on position at the latest time step.
    
    Args:
        past_x_trajs: Array of shape [n_agents, time_steps, 4] where each entry is [px, py, vx, vy]
        top_k: Number of nearest neighbors to return for each agent
        
    Returns:
        mask: Array of shape [n_agents, top_k] containing indices of nearest neighbors
    """
    n_agents = past_x_trajs.shape[0]
    latest_positions = past_x_trajs[:, -1, :2]  
    
    # Compute pairwise squared distances: shape [n_agents, n_agents]
    # Broadcasting: [n_agents, 1, 2] - [1, n_agents, 2] = [n_agents, n_agents, 2]
    diff = latest_positions[:, None, :] - latest_positions[None, :, :]
    squared_distances = jnp.sum(diff ** 2, axis=2)
    
    mask_diag = jnp.eye(n_agents, dtype=bool)
    squared_distances = jnp.where(mask_diag, jnp.inf, squared_distances)
    nearest_indices = jnp.argsort(squared_distances, axis=1)
    mask = nearest_indices[:, :top_k]
    return mask
