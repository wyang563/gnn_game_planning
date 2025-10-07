import jax.numpy as jnp
from models.mlp import MLP

def nearest_neighbors_top_k(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    n_agents = past_x_trajs.shape[0]
    latest_positions = past_x_trajs[:, -1, :2]
    diff = latest_positions[:, None, :] - latest_positions[None, :, :]
    squared_distances = jnp.sum(diff * diff, axis=2)

    mask_diag = jnp.eye(n_agents, dtype=bool)
    squared_distances = jnp.where(mask_diag, jnp.inf, squared_distances)
    
    # Get indices of top-k nearest neighbors for each agent
    nearest_indices = jnp.argsort(squared_distances, axis=1)[:, :top_k]
    
    # Convert indices to mask format
    mask = jnp.zeros((n_agents, n_agents), dtype=jnp.int32)
    row_indices = jnp.arange(n_agents)[:, None]  # Shape: [n_agents, 1]
    mask = mask.at[row_indices, nearest_indices].set(1)
    
    return mask

def jacobian_top_k(
    past_x_trajs: jnp.ndarray,
    top_k: int,
    dt: float,
    w1: float,
    w2: float,
) -> jnp.ndarray:
    N, T, _ = past_x_trajs.shape
    P = past_x_trajs[..., :2]
    
    # calculate gradient relative to position of agent j (pj)
    delta = P[None, :, :] - P[:, None, :]
    d2 = jnp.sum(delta * delta, axis=-1)
    exp_term = jnp.exp(-w2 * d2)
    grad_pj = 2.0 * w1 * w2 * delta * exp_term[..., None] 

    # build position -> control Jacobian matrix
    k = jnp.arange(T)[:, None]
    t = jnp.arange(T - 1)[None, :]
    mask_time = (t < k) & (k > 0)
    coeff = jnp.where(mask_time, ((k - 1 - t) + 0.5) * dt * dt, 0.0)
    coeff = coeff[:, :, None]
    grad_slice = grad_pj[..., 1:, :]
    coeff_slice = coeff[1:, :, :]
    J_sq = jnp.sum(
        (grad_slice[:, :, :, None, :] * coeff_slice[None, None, :, :, :]) ** 2,
        axis=(2,3,4)  # sum over k, t, axis
    )
    scores = jnp.sqrt(J_sq)
    scores = scores * (1.0 - jnp.eye(N))

    # find top k scores per row by indices (highest scores)
    top_k_indices = jnp.argsort(scores, axis=1)[:, -top_k:]
    
    # Convert indices to mask format
    mask = jnp.zeros((N, N), dtype=jnp.int32)
    row_indices = jnp.arange(N)[:, None]  # Shape: [N, 1]
    mask = mask.at[row_indices, top_k_indices].set(1)
    
    return mask

def nearest_neighbors_radius(past_x_trajs: jnp.ndarray, critical_radius: float) -> jnp.ndarray:
    n_agents = past_x_trajs.shape[0]
    latest_positions = past_x_trajs[:, -1, :2]  
    diff = latest_positions[:, None, :] - latest_positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff * diff, axis=2))
    within_radius_mask = distances <= critical_radius
    
    mask_diag = jnp.eye(n_agents, dtype=bool)
    within_radius_mask = jnp.where(mask_diag, False, within_radius_mask)
    return within_radius_mask.astype(jnp.int32)

def run_MLP(flat_x_trajs: jnp.ndarray, model: MLP, n_agents: int) -> jnp.ndarray:
    # Get player masks from model and add in ~0 entries for ego-agent
    player_masks = model(flat_x_trajs)
    final_player_masks = jnp.full((n_agents, n_agents), 1e-5, dtype=jnp.float32)
    rows = jnp.arange(n_agents)[:, None]
    cols = jnp.arange(n_agents - 1)[None, :]
    j_full = jnp.where(cols < rows, cols, cols + 1)
    idx_map = jnp.stack([jnp.broadcast_to(rows, (n_agents, n_agents - 1)), j_full], axis=-1)

    final_player_masks = final_player_masks.at[idx_map[..., 0], idx_map[..., 1]].set(player_masks)

    return final_player_masks

if __name__ == "__main__":
    # Test case for jacobian function
    # Create example past_x_trajs array with 3 agents, 5 time steps, 4 states each
    past_x_trajs = jnp.array([
        # Agent 0: moving from (0,0) to (1,1)
        [[0.0, 0.0, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]],
        # Agent 1: moving from (2,0) to (3,1)  
        [[2.0, 0.0, 0.2, 0.2], [2.2, 0.2, 0.2, 0.2], [2.4, 0.4, 0.2, 0.2], [2.6, 0.6, 0.2, 0.2], [2.8, 0.8, 0.2, 0.2]],
        # Agent 2: moving from (1,2) to (2,3)
        [[1.0, 2.0, 0.2, 0.2], [1.2, 2.2, 0.2, 0.2], [1.4, 2.4, 0.2, 0.2], [1.6, 2.6, 0.2, 0.2], [1.8, 2.8, 0.2, 0.2]]
    ])
    
    # Test parameters
    top_k = 2
    dt = 0.1
    w1 = 1.0
    w2 = 0.5
    
    # Test all neighbor selection functions
    print("Testing neighbor selection functions with example data:")
    print(f"past_x_trajs shape: {past_x_trajs.shape}")
    
    # Test nearest_neighbors_top_k
    print(f"\n1. Nearest neighbors top-k (k={top_k}):")
    nn_mask = nearest_neighbors_top_k(past_x_trajs, top_k)
    print(f"Mask shape: {nn_mask.shape}")
    print(f"Mask:\n{nn_mask}")
    
    # Test jacobian_top_k
    print(f"\n2. Jacobian top-k (k={top_k}):")
    jacobian_mask = jacobian_top_k(past_x_trajs, top_k, dt, w1, w2)
    print(f"Mask shape: {jacobian_mask.shape}")
    print(f"Mask:\n{jacobian_mask}")
    
    # Test nearest_neighbors_radius
    print(f"\n3. Nearest neighbors radius (radius=1.5):")
    radius_mask = nearest_neighbors_radius(past_x_trajs, 1.5)
    print(f"Mask shape: {radius_mask.shape}")
    print(f"Mask:\n{radius_mask}")
    
    # Print final positions for reference
    final_positions = past_x_trajs[:, -1, :2]
    print(f"\nFinal agent positions:")
    for i, pos in enumerate(final_positions):
        print(f"  Agent {i}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Calculate and print actual distances
    print(f"\nPairwise distances:")
    for i in range(3):
        for j in range(i+1, 3):
            dist = jnp.sqrt(jnp.sum((final_positions[i] - final_positions[j])**2))
            print(f"  Agent {i} ↔ Agent {j}: {dist:.2f}")
