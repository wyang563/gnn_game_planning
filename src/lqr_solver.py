import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import LQR
from old_solvers.ilqr_plots import LQRPlotter
import datetime
import os
import random

class Agent(LQR):
    """
    Game-theoretic LQR agent implementing deflected reference trajectory tracking.
    
    This agent implements Option A: single ego agent with discrete-time LQR tracking
    and deflected reference trajectory based on repulsive fields from other agents.
    
    Example usage:
    ```python
    import jax.numpy as jnp
    
    # Define cost weights
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))  # [pos_x, pos_y, vel_x, vel_y]
    R = jnp.eye(2) * 0.01  # [accel_x, accel_y]
    
    # Create agent
    agent = Agent(
        dt=0.01,
        x_dim=4, 
        u_dim=2,
        Q=Q,
        R=R,
        horizon=100,
        x0=jnp.array([0.0, 0.0, 0.0, 0.0]),  # [px, py, vx, vy]
        u0=jnp.array([0.0, 0.0]),  # [ax, ay]
        goal=jnp.array([5.0, 5.0]),  # [target_x, target_y]
        repulsion_gain=1.0,
        repulsion_epsilon=0.1
    )
    
    # Other agents' predicted positions (T, M, 2)
    other_positions = jnp.zeros((100, 2, 2))  # 2 other agents, horizon 100
    
    # Solve for optimal trajectory
    u_traj, x_traj = agent.solve_single_agent(other_positions)
    ```
    """
    def __init__(
        self, 
        dt: float, 
        x_dim: int,
        u_dim: int, 
        Q: jnp.ndarray, 
        R: jnp.ndarray,
        horizon: int,
        x0: jnp.ndarray,
        u0: jnp.ndarray,
        goal: jnp.ndarray,
        device: str = "cpu",
        repulsion_gain: float = 1.0,
        repulsion_epsilon: float = 0.1,
        max_velocity: float = 2.0,
        max_acceleration: float = 2.0,
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)
        self.horizon = horizon
        self.x0 = x0
        self.u0 = u0
        self.goal = goal
        self.device = devices(device)[0]
        self.other_agents = [] # initialized once simulator starts
        
        # Game-theoretic parameters
        self.repulsion_gain = repulsion_gain  # α in the math
        self.repulsion_epsilon = repulsion_epsilon  # ε in the math
        self.max_velocity = max_velocity  # v̄ for velocity clamping
        self.max_acceleration = max_acceleration  # ū for control clamping

        # methods
        self.jit_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.jit_solve = jit(self.solve, device=self.device)
        self.jit_generate_reference_trajectory = jit(self._generate_reference_trajectory, device=self.device)
        self.jit_get_discrete_system_matrices = jit(self.get_discrete_system_matrices, device=self.device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        """
        Continuous-time dynamics: ẋ = f(x, u)
        For double integrator: [px, py, vx, vy] -> [vx, vy, ax, ay]
        This method is used by the LQR linearization process, which handles discretization internally.
        """
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])
    
    def get_u_traj(self):
        # For now initialize intended accelerations to be 0.0, change in the future if needed
        return jnp.tile(jnp.array([0.0, 0.0]), reps=(self.horizon, 1))

    def get_ref_traj(self):
        # p̄_k = p_0 + (k+1)/T * (g - p_0) 
        # But we want to reach the goal at the end, so let's interpolate from k=0 to k=T-1
        # Linear interpolation: start + (k/(T-1)) * (goal - start)
        k_values = jnp.arange(self.horizon)
        p0 = self.x0[:2]
        denom = max(self.horizon - 1, 1)
        return p0[None, :] + (k_values[:, None] / denom) * (self.goal[:2] - p0)[None, :]

    def _generate_reference_trajectory(self, other_agent_positions: jnp.ndarray) -> jnp.ndarray:
        """
        Generate deflected reference trajectory based on game-theoretic formulation.
        
        Args:
            other_agent_positions: Array of shape (T, M, 2) where T is horizon, 
                                 M is number of other agents, last dim is [x, y] position
                                 
        Returns:
            x_ref_traj: Array of shape (T, 4) containing reference states [px, py, vx, vy]
        """
        T = self.horizon
        p0 = self.x0[:2]  # Initial position
        
        baseline_positions = self.get_ref_traj()
        
        # Deflect baseline with smooth repulsive field
        repulsive_forces = jnp.zeros((T, 2))
        
        # Only compute repulsion if there are other agents
        if other_agent_positions.shape[1] > 0:  # M > 0
            # Vectorized computation over all agents and timesteps
            # baseline_positions: (T, 2), other_agent_positions: (T, M, 2)
            # d_jk: (T, M, 2) = baseline_positions[:, None, :] - other_agent_positions
            d_jk = baseline_positions[:, None, :] - other_agent_positions
            
            # r_jk_squared: (T, M)
            r_jk_squared = jnp.sum(d_jk**2, axis=2) + self.repulsion_epsilon
            
            # repulsion_jk: (T, M, 2)
            repulsion_jk = self.repulsion_gain * d_jk / r_jk_squared[:, :, None]**(3/2)
            
            # Sum over all agents: (T, 2)
            repulsive_forces = jnp.sum(repulsion_jk, axis=1)
        
        # p^ref_k = p̄_k + repel_k
        ref_positions = baseline_positions + repulsive_forces
        
        # Step 3: Reference velocity (finite difference)
        # v^ref_0 = (p^ref_0 - p_0) / Δt
        v0 = (ref_positions[0] - p0) / self.dt
        
        # v^ref_k = (p^ref_k - p^ref_{k-1}) / Δt for k ≥ 1
        # Vectorized: diff gives (T-1, 2), so we need to prepend v0
        v_rest = jnp.diff(ref_positions, axis=0) / self.dt
        ref_velocities = jnp.vstack([v0[None, :], v_rest])
        
        # Optional: cap velocity
        ref_velocities = jnp.clip(ref_velocities, -self.max_velocity, self.max_velocity)
        
        # Step 4: Combine into full reference state
        x_ref_traj = jnp.concatenate([ref_positions, ref_velocities], axis=1)
        
        return x_ref_traj

    def get_discrete_system_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate discrete-time system matrices A and B for double integrator.
        
        Returns:
            A: State transition matrix (4x4)
            B: Control input matrix (4x2)
        """
        dt = self.dt
        I2 = jnp.eye(2)
        
        # A = [[I_2, dt*I_2], [0, I_2]]
        A = jnp.block([[I2, dt * I2],
                       [jnp.zeros((2, 2)), I2]])
        
        # B = [[dt^2/2 * I_2], [dt * I_2]]
        B = jnp.block([[0.5 * dt**2 * I2],
                       [dt * I2]])
        
        return A, B

    def solve_single_agent(self, other_agent_positions: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solve the LQR tracking problem for a single agent with deflected reference.
        
        Args:
            other_agent_positions: Array of shape (T, M, 2) containing other agents' predicted positions.
                                 If None, creates empty array (no other agents).
            
        Returns:
            u_traj: Optimal control trajectory (T, 2)
            x_traj: Optimal state trajectory (T+1, 4)
        """
        # Handle case with no other agents
        if other_agent_positions is None:
            other_agent_positions = jnp.zeros((self.horizon, 0, 2))
        
        # Generate deflected reference trajectory (T timesteps)
        x_ref_traj = self.jit_generate_reference_trajectory(other_agent_positions)
        
        # Use linearize_dyn like in the notebook example to get proper A,B matrices
        dummy_u_traj = jnp.zeros((self.horizon, self.u_dim))
        _, A_traj, B_traj = self.jit_linearize_dyn(self.x0, dummy_u_traj)
        
        # Solve LQR tracking problem
        # The LQR class expects reference trajectory, not differences
        u_traj, x_traj = self.jit_solve(self.x0, A_traj, B_traj, x_ref_traj)
        
        # Apply control limits
        u_traj = jnp.clip(u_traj, -self.max_acceleration, self.max_acceleration)
        
        return u_traj, x_traj

def random_init(n_agents: int, 
                init_position_range: Tuple[float, float]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    init_ps = []
    init_us = []
    goals = []
    
    min_pos, max_pos = init_position_range
    pos_range = max_pos - min_pos
    
    # Minimum distance between agents to avoid initial collisions
    min_distance = 0.5 * pos_range / n_agents  # Scale with number of agents
    
    max_tries = 1000
    
    for _ in range(n_agents):
        # Generate initial position
        init_pos = None
        for _ in range(max_tries):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            theta = random.uniform(-jnp.pi, jnp.pi)
            
            candidate_pos = jnp.array([x, y, theta])
            
            # Check minimum distance from other agents
            too_close = False
            for existing_pos in init_ps:
                distance = jnp.linalg.norm(candidate_pos[:2] - existing_pos[:2])
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                init_pos = candidate_pos
                break
        
        init_ps.append(init_pos)
        
        # Generate initial control (random velocity and angular velocity)
        v = random.uniform(0.3, 1.2)  # Linear velocity
        omega = random.uniform(-0.5, 0.5)  # Angular velocity
        init_us.append(jnp.array([v, omega]))
        
        # Generate goal position (different from initial position)
        goal = None
        for _ in range(max_tries):
            goal_x = random.uniform(min_pos, max_pos)
            goal_y = random.uniform(min_pos, max_pos)
            candidate_goal = jnp.array([goal_x, goal_y])
            
            # Ensure goal is far enough from initial position
            distance_to_start = jnp.linalg.norm(candidate_goal - init_pos[:2])
            if distance_to_start > min_distance:
                goal = candidate_goal
                break
        
        goals.append(goal)
    
    return init_ps, init_us, goals

if __name__ == "__main__":
    # Simple multi-agent demo: create N agents, compute trajectories, and check convergence
    N = 3
    horizon = 40
    dt = 0.1

    # Reproducibility
    random.seed(42)

    # Cost weights
    Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))
    R = jnp.eye(2) * 0.1

    # Random initial states/controls/goals
    init_ps, init_us, goals = random_init(N, (-5.0, 5.0))

    # Build agents
    agents = []
    for i in range(N):
        px, py, theta = float(init_ps[i][0]), float(init_ps[i][1]), float(init_ps[i][2])
        v_i, _omega_i = float(init_us[i][0]), float(init_us[i][1])
        vx_i = v_i * jnp.cos(theta)
        vy_i = v_i * jnp.sin(theta)

        x0 = jnp.array([px, py, vx_i, vy_i])
        u0 = jnp.array([0.0, 0.0])
        goal_i = goals[i]

        agent = Agent(
            dt=dt,
            x_dim=4,
            u_dim=2,
            Q=Q,
            R=R,
            horizon=horizon,
            x0=x0,
            u0=u0,
            goal=goal_i,
            device="cpu",
            repulsion_gain=1.0,
            repulsion_epsilon=0.1,
            max_velocity=2.0,
            max_acceleration=2.0,
        )
        agents.append(agent)

    # Precompute straight-line predicted positions for other agents as baseline forecasts
    k_values = jnp.arange(horizon)
    denom = max(horizon - 1, 1)
    straight_line_preds = []  # List of (T, 2) arrays for each agent's baseline path
    for j in range(N):
        p0_j = init_ps[j][:2]
        g_j = goals[j]
        baseline_j = p0_j[None, :] + (k_values[:, None] / denom) * (g_j[None, :] - p0_j[None, :])
        straight_line_preds.append(baseline_j)

    # Solve for each agent using others' baseline predictions
    results = []
    for i, agent in enumerate(agents):
        other_positions_list = []
        for j in range(N):
            if j == i:
                continue
            other_positions_list.append(straight_line_preds[j])
        if len(other_positions_list) == 0:
            other_agent_positions = jnp.zeros((horizon, 0, 2))
        else:
            other_agent_positions = jnp.stack(other_positions_list, axis=1)  # (T, M, 2)

        u_traj, x_traj = agent.solve_single_agent(other_agent_positions)
        results.append((u_traj, x_traj))

    # Check convergence: final position close to goal
    tol = 0.2
    num_converged = 0
    for i, (u_traj, x_traj) in enumerate(results):
        final_pos = x_traj[-1, :2]
        dist = float(jnp.linalg.norm(final_pos - goals[i]))
        converged = dist < tol
        num_converged += int(converged)
        print(f"Agent {i}: final_distance_to_goal={dist:.3f}, converged={converged}")

    print(f"Converged {num_converged}/{N} agents within tolerance {tol}.")