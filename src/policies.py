import jax.numpy as jnp
from jax import jit, vmap, grad, lax

def nearest_neighbors(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    """
    Find the top_k nearest neighbors for each agent based on position at the latest time step.
    
    Args:
        past_x_trajs: Array of shape [n_agents, time_steps, 4] where each entry is [px, py, vx, vy]
        top_k: Number of nearest neighbors to return for each agent (treated as static for JIT)
        
    Returns:
        indices: Array of shape [n_agents, top_k] containing indices of nearest neighbors
    """
    n_agents = past_x_trajs.shape[0]
    latest_positions = past_x_trajs[:, -1, :2]
    diff = latest_positions[:, None, :] - latest_positions[None, :, :]
    squared_distances = jnp.sum(diff * diff, axis=2)

    mask_diag = jnp.eye(n_agents, dtype=bool)
    squared_distances = jnp.where(mask_diag, jnp.inf, squared_distances)
    # Full sort, then take the first top_k columns (top_k is static for JIT)
    nearest_indices = jnp.argsort(squared_distances, axis=1)
    return nearest_indices[:, :top_k]

def jacobian(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    pass

def cost_evolution(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    pass
