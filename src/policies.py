import jax.numpy as jnp
from jax import jit, vmap, grad, lax

def nearest_neighbors(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    n_agents = past_x_trajs.shape[0]
    latest_positions = past_x_trajs[:, -1, :2]
    diff = latest_positions[:, None, :] - latest_positions[None, :, :]
    squared_distances = jnp.sum(diff * diff, axis=2)

    mask_diag = jnp.eye(n_agents, dtype=bool)
    squared_distances = jnp.where(mask_diag, jnp.inf, squared_distances)
    # Full sort, then take the first top_k columns (top_k is static for JIT)
    nearest_indices = jnp.argsort(squared_distances, axis=1)
    return nearest_indices[:, :top_k]

def jacobian(
    past_x_trajs: jnp.ndarray,
    top_k: int,
    dt: float,
    w1: float,
    w2: float,
) -> jnp.ndarray:
    # TODO: debug later
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
    mask = (t < k) & (k > 0)
    coeff = jnp.where(mask, ((k - 1 - t) + 0.5) * dt * dt, 0.0)
    coeff = coeff[:, :, None]
    grad_slice = grad_pj[..., 1:, :]
    coeff_slice = coeff[1:, :, :]
    J_sq = jnp.sum(
        (grad_slice[:, :, :, None, :] * coeff_slice[None, None, :, :, :]) ** 2,
        axis=(2,3,4)  # sum over k, t, axis
    )
    scores = jnp.sqrt(J_sq)
    scores = scores * (1.0 - jnp.eye(N))

    # find top k scores per row by indices
    top_k_indices = jnp.argsort(scores, axis=1)[:, -top_k:]
    return top_k_indices

def cost_evolution(past_x_trajs: jnp.ndarray, top_k: int) -> jnp.ndarray:
    pass

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
    
    # Call jacobian function
    print("Testing jacobian function with example data:")
    print(f"past_x_trajs shape: {past_x_trajs.shape}")
    jacobian(past_x_trajs, top_k, dt, w1, w2)
