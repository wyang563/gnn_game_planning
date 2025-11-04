import jax.numpy as jnp


def nearest_neighbors(x_trajs: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Select k nearest neighbors to the ego agent (agent 0) based on Euclidean distance.
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim) where state_dim >= 2 (at least x, y positions).
        k: Number of nearest neighbors to select.
    
    Returns:
        A binary mask array of shape (N,) where N is the number of agents.
        Indices with value 1 correspond to the k nearest neighbors (excluding ego at index 0).
    """
    if x_trajs.ndim == 2:
        # Handle flattened input (N_agents, T*state_dim) - reshape to (N_agents, T, state_dim)
        N = x_trajs.shape[0]
        # Assume state_dim=4 (x, y, vx, vy)
        state_dim = 4
        T = x_trajs.shape[1] // state_dim
        x_trajs = x_trajs.reshape(N, T, state_dim)
    
    N, T, state_dim = x_trajs.shape
    mask = jnp.zeros(N)
    
    # Handle edge cases
    if T < 1 or k <= 0:
        return mask
    
    ego_pos = x_trajs[0, -1, :2]
    other_pos = x_trajs[1:, -1, :2]
    
    distances = jnp.linalg.norm(ego_pos - other_pos, axis=1)
    ranked_indices = jnp.argsort(distances)
    num_neighbors = max(0, min(k, len(ranked_indices)))
    top_indices = ranked_indices[:num_neighbors]
    top_indices = top_indices + 1  # increment by 1 to account for ego agent
    mask = mask.at[top_indices].set(1)
    return mask


def jacobian(x_trajs: jnp.ndarray, k: int, delta_t: float = 0.1) -> jnp.ndarray:
    """
    Select k agents with highest Jacobian norm of the proximity cost function.
    
    The cost function is exp(-D) where D is the squared norm of the predicted state difference.
    The Jacobian measures sensitivity of this cost to velocity differences (approximated from trajectory).
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim) where state_dim >= 4 (x, y, vx, vy).
        k: Number of agents to select.
        delta_t: Time step for prediction (default: 0.1).
    
    Returns:
        A binary mask array of shape (N,) where N is the number of agents.
        Indices with value 1 correspond to the k agents with highest Jacobian norm.
    """
    if x_trajs.ndim == 2:
        # Handle flattened input (N_agents, T*state_dim) - reshape to (N_agents, T, state_dim)
        N = x_trajs.shape[0]
        state_dim = 4
        T = x_trajs.shape[1] // state_dim
        x_trajs = x_trajs.reshape(N, T, state_dim)
    
    N, T, state_dim = x_trajs.shape
    mask = jnp.zeros(N)
    
    # Handle edge cases
    if T < 2 or k <= 0 or state_dim < 4:
        return mask
    
    norm_costs = jnp.zeros(N - 1)
    ego_state = x_trajs[0, -1, :]  # Last state
    
    for i in range(N - 1):
        player_id = i + 1
        
        # State difference between ego and other agent
        state_diff = ego_state - x_trajs[player_id, -1, :]
        
        # Estimate control from velocity change (a = dv/dt)
        if T >= 2:
            vel_change = x_trajs[player_id, -1, 2:4] - x_trajs[player_id, -2, 2:4]
            estimated_control = vel_change / delta_t
        else:
            estimated_control = jnp.zeros(2)
        
        # Predicted state differences
        delta_px = (state_diff[0] + delta_t * state_diff[2]) ** 2
        delta_py = (state_diff[1] + delta_t * state_diff[3]) ** 2
        delta_vx = (state_diff[2] + delta_t * estimated_control[0]) ** 2
        delta_vy = (state_diff[3] + delta_t * estimated_control[1]) ** 2
        
        # Total squared distance
        D = delta_px + delta_py + delta_vx + delta_vy
        
        if D > 1e-10:  # Avoid division by zero
            # Future velocity differences
            future_vel_diff_x = state_diff[2] + delta_t * estimated_control[0]
            future_vel_diff_y = state_diff[3] + delta_t * estimated_control[1]
            
            # Jacobian of cost = exp(-D) with respect to control
            exp_term = jnp.exp(-D)
            J1 = 2 * delta_t * future_vel_diff_x * exp_term
            J2 = 2 * delta_t * future_vel_diff_y * exp_term
            norm_costs = norm_costs.at[i].set(jnp.linalg.norm(jnp.array([J1, J2])))
        else:
            norm_costs = norm_costs.at[i].set(0.0)
    
    # Rank by Jacobian norm (highest to lowest)
    ranked_indices = jnp.argsort(norm_costs)[::-1]
    num_neighbors = max(0, min(k, len(ranked_indices)))
    top_indices = ranked_indices[:num_neighbors]
    top_indices = top_indices + 1  # increment by 1 to account for ego agent
    
    mask = mask.at[top_indices].set(1)
    
    return mask


def cost_evolution(x_trajs: jnp.ndarray, k: int, mu: float = 1.0) -> jnp.ndarray:
    """
    Select k agents with highest cost evolution (increasing proximity cost).
    
    Cost evolution measures how much the proximity cost is changing over time.
    Higher values indicate agents that are getting closer (more important to consider).
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim) where state_dim >= 2 (x, y) and T >= 2.
        k: Number of agents to select.
        mu: Scaling factor for the cost (default: 1.0).
    
    Returns:
        A binary mask array of shape (N,) where N is the number of agents.
        Indices with value 1 correspond to the k agents with highest cost evolution.
    """
    if x_trajs.ndim == 2:
        # Handle flattened input (N_agents, T*state_dim) - reshape to (N_agents, T, state_dim)
        N = x_trajs.shape[0]
        state_dim = 4
        T = x_trajs.shape[1] // state_dim
        x_trajs = x_trajs.reshape(N, T, state_dim)
    
    N, T, state_dim = x_trajs.shape
    mask = jnp.zeros(N)
    
    # Handle edge cases - need at least 2 timesteps
    if T < 2 or k <= 0:
        return mask
    
    cost_evolution_values = jnp.zeros(N - 1)
    
    for i in range(N - 1):
        player_id = i + 1
        
        # Current state difference (position only)
        state_diff = x_trajs[0, -1, :2] - x_trajs[player_id, -1, :2]
        D = jnp.sum(state_diff**2)
        
        # Previous state difference (position only)
        state_diff_prev = x_trajs[0, -2, :2] - x_trajs[player_id, -2, :2]
        D_prev = jnp.sum(state_diff_prev**2)
        
        if D > 1e-10 and D_prev > 1e-10:  # Avoid numerical issues
            # Evolution of cost: current cost - previous cost
            cost_evolution_values = cost_evolution_values.at[i].set(mu * jnp.exp(-D) - mu * jnp.exp(-D_prev))
        else:
            cost_evolution_values = cost_evolution_values.at[i].set(0.0)
    
    # Rank by cost evolution (highest to lowest)
    # Higher values mean cost is increasing (agents getting closer)
    ranked_indices = jnp.argsort(cost_evolution_values)[::-1]
    num_neighbors = max(0, min(k, len(ranked_indices)))
    top_indices = ranked_indices[:num_neighbors]
    top_indices = top_indices + 1  # increment by 1 to account for ego agent

    mask = mask.at[top_indices].set(1)
    
    return mask


def barrier_function(x_trajs: jnp.ndarray, k: int, R: float = 0.5, kappa: float = 5.0) -> jnp.ndarray:
    """
    Select k agents with smallest barrier function values (most dangerous/closest to constraint violation).
    
    The barrier function measures safety constraints based on inter-agent distances.
    Lower values indicate agents that are closer to violating safety constraints.
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim) where state_dim >= 4 (x, y, vx, vy).
        k: Number of agents to select.
        R: Safety radius (default: 0.5).
        kappa: Barrier function gain (default: 5.0).
    
    Returns:
        A binary mask array of shape (N,) where N is the number of agents.
        Indices with value 1 correspond to the k agents with smallest barrier function values
        (most dangerous agents).
    """
    if x_trajs.ndim == 2:
        # Handle flattened input (N_agents, T*state_dim) - reshape to (N_agents, T, state_dim)
        N = x_trajs.shape[0]
        state_dim = 4
        T = x_trajs.shape[1] // state_dim
        x_trajs = x_trajs.reshape(N, T, state_dim)
    
    N, T, state_dim = x_trajs.shape
    mask = jnp.zeros(N)
    
    # Handle edge cases
    if T < 1 or k <= 0 or state_dim < 4:
        return mask
    
    bf_values = jnp.zeros(N - 1)
    
    for i in range(N - 1):
        player_id = i + 1
        
        # Position difference (x, y)
        pos_diff = x_trajs[0, -1, :2] - x_trajs[player_id, -1, :2]
        
        # Velocity difference (vx, vy)
        vel_diff = x_trajs[0, -1, 2:4] - x_trajs[player_id, -1, 2:4]
        
        # Barrier function: h = ||p||^2 - R^2
        h = jnp.sum(pos_diff**2) - R**2
        
        # Derivative: h_dot = 2 * (p^T * v)
        h_dot = 2 * jnp.dot(pos_diff, vel_diff)
        
        # Control barrier function condition: h_dot + kappa * h >= 0
        # Lower values are more dangerous
        bf_values = bf_values.at[i].set(h_dot + kappa * h)
    
    # Rank from smallest to largest (smallest = most dangerous)
    ranked_indices = jnp.argsort(bf_values)
    num_neighbors = max(0, min(k, len(ranked_indices)))
    top_indices = ranked_indices[:num_neighbors]
    top_indices = top_indices + 1  # increment by 1 to account for ego agent
    mask = mask.at[top_indices].set(1)

    return mask

