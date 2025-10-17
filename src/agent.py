import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

class Agent(iLQR):
    def __init__(
        self, 
        id: int,
        dt: float, 
        x_dim: int,
        u_dim: int, 
        Q: jnp.ndarray, 
        R: jnp.ndarray,
        W: jnp.ndarray,
        horizon: int,
        time_steps: int,
        x0: jnp.ndarray,
        u_traj: jnp.ndarray,
        goal: jnp.ndarray,
        device = None,
        goal_threshold: float = 0.1,
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)
        # simulation parameters 
        self.id = id
        self.horizon = horizon
        self.time_steps = time_steps
        # Prefer CUDA if available; fallback to CPU
        if device is None:
            cuda_devs = jax.devices("cuda")
            self.device = cuda_devs[0] if len(cuda_devs) > 0 else jax.devices("cpu")[0]
        else:
            self.device = device

        # Place parameters and state on target device once to avoid repeated transfers
        W_dev = jax.device_put(W, self.device)
        self.w1, self.w2, self.w3, self.w4 = W_dev[0], W_dev[1], W_dev[2], W_dev[3]
        self.x0 = jax.device_put(x0, self.device)
        self.u_traj = jax.device_put(u_traj, self.device)
        self.goal = jax.device_put(goal, self.device)
        self.ref_traj = jax.device_put(self.get_ref_traj(), self.device)
        self.goal_threshold = goal_threshold
        self.x_traj: Optional[jnp.ndarray] = None  

        # values for logging
        self.loss_history: List[float] = []

        # methods
        self.jit_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.jit_solve = jit(self.solve, device=self.device)
        self.jit_loss = jit(self.loss, device=self.device)
        self.jit_linearize_loss = jit(self.linearize_loss, device=self.device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])
    
    # define loss functions
    def runtime_loss(self, xt, ut, ref_xt, other_xts, mask):
        # calculate collision loss
        current_position = xt[:2]
        squared_distances = jnp.sum(jnp.square(current_position - other_xts), axis=1)
        collision_loss = self.w1 * jnp.exp(-self.w2 * squared_distances)
        # Apply mask to collision loss, not distance
        # differentiably clamp values to close to 0 or 1 using formula: 0.5 * (1 + tanh(10 * (mask - 0.5)))
        # eps = 1e-8
        alpha = 16.0
        modified_mask = jax.nn.sigmoid(alpha * (mask - 0.5))
        collision_loss = collision_loss * modified_mask 
        # Normalize by number of active agents (sum of mask) instead of total agents
        # normalizer = jax.lax.stop_gradient(jnp.sum(modified_mask) + eps)
        collision_loss = jnp.sum(collision_loss) 

        ctrl_loss: jnp.ndarray = self.w3 * jnp.sum(jnp.square(ut))
        nav_loss: jnp.ndarray = self.w4 * jnp.sum(jnp.square(xt[:2]-ref_xt[:2]))
        return nav_loss + collision_loss + ctrl_loss

    def loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
        runtime_loss_array = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask)
        return runtime_loss_array.sum() * self.dt

    def linearize_loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj = vmap(dldx, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask)  
        b_traj = vmap(dldu, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask) 
        return a_traj, b_traj

    # helpers
    def debug_agent(self):
        print(f"Agent {self.id}:")
        print(f"  x0={self.x0}")
        print(f"  goal={self.goal}")
        print(f"  ref_traj={self.ref_traj}")
        print(f"  x_traj={self.x_traj}")
        print(f"  u_traj={self.u_traj}")

    # whole simulation versions of function
    def get_ref_traj(self):
        return jnp.linspace(self.x0[:2], self.goal, self.time_steps+1)[1:]
    
    def calculate_x_traj(self):
        return self.jit_linearize_dyn(self.x0, self.u_traj)[0]

    def check_convergence(self) -> bool:
        final_x_pos = self.calculate_x_traj()[-1, :2]
        return jnp.linalg.norm(final_x_pos - self.goal) < self.goal_threshold


if __name__ == "__main__":
    # Quick manual test for runtime_loss
    dt = 0.1
    x_dim = 4
    u_dim = 2
    Q = jnp.eye(x_dim)
    R = jnp.eye(u_dim)
    W = jnp.array([1.0, 1.0, 0.1, 0.5], dtype=jnp.float32)
    horizon = 5
    time_steps = 5
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    u_traj = jnp.zeros((time_steps, u_dim), dtype=jnp.float32)
    goal = jnp.array([1.0, 1.0], dtype=jnp.float32)

    agent = Agent(
        id=0,
        dt=dt,
        x_dim=x_dim,
        u_dim=u_dim,
        Q=Q,
        R=R,
        W=W,
        horizon=horizon,
        time_steps=time_steps,
        x0=x0,
        u_traj=u_traj,
        goal=goal,
        device=None,
    )

    xt = jnp.array([0.2, 0.3, 0.0, 0.0], dtype=jnp.float32)
    ut = jnp.array([0.0, 0.0], dtype=jnp.float32)
    ref_xt = jnp.array([0.25, 0.35, 0.0, 0.0], dtype=jnp.float32)
    other_xts = jnp.array(
        [
            [0.2, 0.3],   # very close
            [1.2, 1.3],   # moderate distance
            [10.0, 10.0], # far away
        ],
        dtype=jnp.float32,
    )
    mask = jnp.array([0.9, 0.11, 0.87], dtype=jnp.float32)

    # Run without JIT for clearer debugging behavior
    with jax.disable_jit():
        loss_val = agent.runtime_loss(xt, ut, ref_xt, other_xts, mask)
        print("runtime_loss (manual test):", loss_val)
