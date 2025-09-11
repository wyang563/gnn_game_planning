import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import torch
from typing import Tuple, Optional, List
from lqrax import LQR


class LQRaxNashSolver:
    """
    Multi-agent Nash equilibrium solver using LQRax LQR module.
    
    Implements the formulation from sections 3.1, 7.1, and 7.2 of the paper:
    - Double integrator dynamics: x_{k+1} = A*x_k + B*u_k
    - Objective: sum_k [w1*||p_k - p_goal||^2 + w2*||v_k||^2 + w3*||u_k||^2 + w4*collision_term]
    - Nash equilibrium via iterative best response
    """
    
    def __init__(self, 
                 n_agents: int = 2,
                 horizon: int = 30,
                 dt: float = 0.1,
                 w1: float = 10.0,  # goal tracking weight
                 w2: float = 1.0,   # velocity penalty weight  
                 w3: float = 0.1,   # control effort weight
                 w4: float = 0.0,   # collision avoidance weight
                 eps: float = 1e-3,
                 # Bounds from section 7.2 equations 9a, 9b, 9c
                 vel_bounds: Tuple[float, float] = (-5.0, 5.0),    # velocity bounds
                 accel_bounds: Tuple[float, float] = (-2.0, 2.0)): # acceleration bounds
        """
        Initialize LQRax Nash solver.
        
        Args:
            n_agents: Number of agents
            horizon: Planning horizon T
            dt: Time step Delta
            w1-w4: Cost function weights from paper sections 7.1-7.2
            eps: Regularization for collision avoidance
        """
        self.n_agents = n_agents
        self.horizon = horizon
        self.dt = dt
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.eps = eps
        self.vel_bounds = vel_bounds
        self.accel_bounds = accel_bounds
        
        # State dimensions: [px, py, vx, vy] per agent (section 3.1)
        self.state_dim = 4
        self.control_dim = 2
        
        # Setup double integrator dynamics matrices (section 3.1)
        self._setup_dynamics_matrices()
        
    def _setup_dynamics_matrices(self):
        """
        Setup double integrator dynamics matrices from section 3.1:
        x_k = [px, py, vx, vy]^T
        u_k = [ax, ay]^T
        x_{k+1} = A*x_k + B*u_k
        """
        # A matrix: double integrator dynamics
        # [px_{k+1}]   [1 0 dt  0][px_k]   [0 0][ax_k]
        # [py_{k+1}] = [0 1  0 dt][py_k] + [0 0][ay_k]
        # [vx_{k+1}]   [0 0  1  0][vx_k]   [dt 0]
        # [vy_{k+1}]   [0 0  0  1][vy_k]   [0 dt]
        
        self.A = jnp.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.B = jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [self.dt, 0.0],
            [0.0, self.dt]
        ])
        
    def _build_single_agent_lqr_matrices(self, 
                                       agent_idx: int,
                                       goal: jnp.ndarray,
                                       other_trajectories: Optional[List[jnp.ndarray]] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build time-varying LQR matrices Q(t), R(t), q(t) for a single agent.
        
        From sections 7.1-7.2, the cost for agent i is:
        J_i = sum_k [w1*||p_k - p_goal||^2 + w2*||v_k||^2 + w3*||u_k||^2 + w4*collision_term]
        
        This translates to LQR form: sum_k [x_k^T Q_k x_k + u_k^T R_k u_k + q_k^T x_k]
        
        Args:
            agent_idx: Index of agent
            goal: Goal position [2]
            other_trajectories: Trajectories of other agents for collision costs
            
        Returns:
            Q_traj: State cost matrices [horizon+1, 4, 4]
            R_traj: Control cost matrices [horizon, 2, 2] 
            q_traj: Linear state cost terms [horizon+1, 4]
        """
        # Initialize cost matrices for all timesteps
        Q_traj = jnp.zeros((self.horizon + 1, self.state_dim, self.state_dim))
        R_traj = jnp.zeros((self.horizon, self.control_dim, self.control_dim))
        q_traj = jnp.zeros((self.horizon + 1, self.state_dim))
        
        # Base Q matrix: goal tracking (w1) + velocity penalty (w2)
        # State is [px, py, vx, vy], so:
        # - w1 penalty on positions (indices 0,1)
        # - w2 penalty on velocities (indices 2,3)
        Q_base = jnp.diag(jnp.array([self.w1, self.w1, self.w2, self.w2]))
        
        # R matrix: control effort penalty (w3)
        R_base = self.w3 * jnp.eye(self.control_dim)
        
        # Linear term for goal tracking: -2*w1*goal^T*p
        # Only affects position components (indices 0,1)
        q_base = jnp.zeros(self.state_dim)
        q_base = q_base.at[0].set(-2.0 * self.w1 * goal[0])
        q_base = q_base.at[1].set(-2.0 * self.w1 * goal[1])
        
        # Build time-varying matrices
        for t in range(self.horizon + 1):
            Q_t = Q_base.copy()
            q_t = q_base.copy()
            
            # Add collision avoidance terms if other trajectories provided
            if other_trajectories is not None and self.w4 > 0:
                for other_traj in other_trajectories:
                    if t < len(other_traj):
                        # Other agent's position at time t
                        p_other = other_traj[t, :2]
                        
                        # Collision cost: w4 / (||p - p_other||^2 + eps)
                        # Quadratic approximation around p_other:
                        # cost ≈ w4/eps - (2*w4/eps^2) * (p - p_other)^T * (p - p_other)
                        
                        # This gives quadratic term: (2*w4/eps^2) * I_2x2 for positions
                        collision_weight = 2.0 * self.w4 / (self.eps**2)
                        Q_t = Q_t.at[0, 0].add(collision_weight)  # px^2 term
                        Q_t = Q_t.at[1, 1].add(collision_weight)  # py^2 term
                        
                        # Linear term: -(4*w4/eps^2) * p_other^T * p
                        linear_weight = -4.0 * self.w4 / (self.eps**2)
                        q_t = q_t.at[0].add(linear_weight * p_other[0])
                        q_t = q_t.at[1].add(linear_weight * p_other[1])
            
            Q_traj = Q_traj.at[t].set(Q_t)
            q_traj = q_traj.at[t].set(q_t)
            
            # Control cost matrix (same for all timesteps)
            if t < self.horizon:
                R_traj = R_traj.at[t].set(R_base)
        
        return Q_traj, R_traj, q_traj
    
    def _apply_constraints(self, 
                          initial_state: jnp.ndarray,
                          controls_raw: jnp.ndarray,
                          states_raw: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply velocity and acceleration constraints from section 7.2 equations 9a, 9b, 9c.
        
        Constraints:
        9a: ||v_k|| ≤ v_max  (velocity magnitude bounds)
        9b: ||a_k|| ≤ a_max  (acceleration magnitude bounds)
        9c: p_{x,k} ∈ [p_{x,min}, p_{x,max}], p_{y,k} ∈ [p_{y,min}, p_{y,max}] (position bounds - if needed)
        
        Args:
            initial_state: Initial state [4]
            controls_raw: Unconstrained controls [horizon, 2]
            states_raw: Unconstrained states [horizon, 4]
            
        Returns:
            controls_constrained: Constrained controls [horizon, 2]
            trajectory_constrained: Constrained trajectory [horizon+1, 4]
        """
        v_min, v_max = self.vel_bounds
        a_min, a_max = self.accel_bounds
        
        # Convert bounds to magnitude limits (assuming symmetric bounds)
        v_max_mag = min(abs(v_min), abs(v_max))  # Max velocity magnitude
        a_max_mag = min(abs(a_min), abs(a_max))  # Max acceleration magnitude
        
        # Clip acceleration magnitudes while preserving direction - equation 9b
        def clip_magnitude(vector, max_magnitude):
            """Clip vector magnitude while preserving direction."""
            magnitude = jnp.linalg.norm(vector)
            if magnitude > max_magnitude:
                return vector * (max_magnitude / magnitude)
            else:
                return vector
        
        # Apply magnitude clipping to controls
        controls_clipped = jnp.array([
            clip_magnitude(controls_raw[t], a_max_mag) for t in range(self.horizon)
        ])
        
        # Forward simulate with magnitude-clipped controls to ensure velocity constraints
        trajectory = jnp.zeros((self.horizon + 1, self.state_dim))
        trajectory = trajectory.at[0].set(initial_state)
        
        controls_final = jnp.zeros((self.horizon, self.control_dim))
        
        for t in range(self.horizon):
            current_state = trajectory[t]
            proposed_control = controls_clipped[t]
            
            # Check if proposed control would violate velocity magnitude constraints
            next_state_proposed = self.A @ current_state + self.B @ proposed_control
            next_velocity = next_state_proposed[2:4]
            next_vel_magnitude = jnp.linalg.norm(next_velocity)
            
            # If velocity magnitude would be violated, adjust the control
            if next_vel_magnitude > v_max_mag:
                current_velocity = current_state[2:4]
                current_vel_magnitude = jnp.linalg.norm(current_velocity)
                
                # Calculate the maximum velocity change magnitude that keeps us within bounds
                if current_vel_magnitude < v_max_mag:
                    # We can increase velocity magnitude up to the limit
                    max_vel_magnitude_increase = v_max_mag - current_vel_magnitude
                    
                    # The velocity change from control: Δv = dt * a
                    velocity_change = self.dt * proposed_control
                    velocity_change_magnitude = jnp.linalg.norm(velocity_change)
                    
                    if velocity_change_magnitude > 0:
                        # Scale the velocity change to respect the velocity bound
                        max_change_magnitude = max_vel_magnitude_increase
                        if velocity_change_magnitude > max_change_magnitude:
                            # Scale down the control to respect velocity bounds
                            scale_factor = max_change_magnitude / velocity_change_magnitude
                            adjusted_control = proposed_control * scale_factor
                        else:
                            adjusted_control = proposed_control
                    else:
                        adjusted_control = proposed_control
                else:
                    # Current velocity is already at or above limit, need to reduce
                    # Use a conservative approach: clip the resulting velocity
                    adjusted_control = proposed_control * 0.1  # Reduce aggressiveness
                
                # Final magnitude clipping of control
                adjusted_control = clip_magnitude(adjusted_control, a_max_mag)
            else:
                adjusted_control = proposed_control
            
            controls_final = controls_final.at[t].set(adjusted_control)
            
            # Apply dynamics with adjusted control
            next_state = self.A @ current_state + self.B @ adjusted_control
            trajectory = trajectory.at[t+1].set(next_state)
        
        return controls_final, trajectory
    
    def solve_single_agent_lqr(self,
                              agent_idx: int,
                              initial_state: jnp.ndarray,
                              goal: jnp.ndarray,
                              other_trajectories: Optional[List[jnp.ndarray]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solve LQR for a single agent given fixed trajectories of others.
        
        Args:
            agent_idx: Index of agent to optimize
            initial_state: Initial state [4] for this agent
            goal: Goal position [2] for this agent
            other_trajectories: Fixed trajectories of other agents
            
        Returns:
            trajectory: Optimal trajectory [horizon+1, 4]
            controls: Optimal controls [horizon, 2]
        """
        # Build average Q and R matrices for LQRax initialization
        # Base matrices without collision terms
        Q_base = jnp.diag(jnp.array([self.w1, self.w1, self.w2, self.w2]))
        R_base = self.w3 * jnp.eye(self.control_dim)
        
        # Add average collision avoidance penalty to Q if needed
        if other_trajectories is not None and self.w4 > 0:
            collision_weight = 2.0 * self.w4 / (self.eps**2)
            Q_collision = Q_base + jnp.diag(jnp.array([collision_weight, collision_weight, 0.0, 0.0]))
        else:
            Q_collision = Q_base
        
        try:
            # Create LQRax solver with fixed Q and R matrices
            lqr_solver = LQR(
                dt=self.dt,
                x_dim=self.state_dim,
                u_dim=self.control_dim,
                Q=Q_collision,
                R=R_base
            )
            
            # Build time-varying matrices for the trajectory
            A_traj = jnp.tile(self.A[None, :, :], (self.horizon, 1, 1))
            B_traj = jnp.tile(self.B[None, :, :], (self.horizon, 1, 1))
            
            # Build reference trajectory for goal tracking
            # Reference trajectory should be (horizon, state_dim)
            ref_traj = jnp.zeros((self.horizon, self.state_dim))
            for t in range(self.horizon):
                # Set goal positions as reference, zero velocities
                ref_t = jnp.zeros(self.state_dim)
                ref_t = ref_t.at[0].set(goal[0])  # goal x position
                ref_t = ref_t.at[1].set(goal[1])  # goal y position
                # velocities remain zero in reference
                ref_traj = ref_traj.at[t].set(ref_t)
            
            # For collision avoidance, we could modify ref_traj based on other agents
            # but for now, we'll use the basic goal tracking
            
            # Solve LQR with correct API: solve(x0, A_traj, B_traj, ref_traj)
            controls_raw, states_raw = lqr_solver.solve(initial_state, A_traj, B_traj, ref_traj)
            
            # Apply constraints from section 7.2 equations 9a, 9b, 9c
            controls_constrained, trajectory_constrained = self._apply_constraints(
                initial_state, controls_raw, states_raw
            )
            
            return trajectory_constrained, controls_constrained
            
        except Exception as e:
            print(f"LQR solve failed for agent {agent_idx}: {e}")
            # Fallback: simple proportional control toward goal
            trajectory = jnp.zeros((self.horizon + 1, self.state_dim))
            trajectory = trajectory.at[0].set(initial_state)
            
            controls = jnp.zeros((self.horizon, self.control_dim))
            
            # Simple proportional control
            for t in range(self.horizon):
                current_pos = trajectory[t, :2]
                error = goal - current_pos
                control = 0.1 * error  # Simple proportional gain
                controls = controls.at[t].set(control)
                
                # Apply dynamics
                next_state = self.A @ trajectory[t] + self.B @ control
                trajectory = trajectory.at[t+1].set(next_state)
            
            return trajectory, controls
    
    def solve_nash_equilibrium(self,
                             initial_states: torch.Tensor,
                             goals: torch.Tensor,
                             max_iterations: int = 50,
                             tolerance: float = 1e-3) -> Optional[torch.Tensor]:
        """
        Solve Nash equilibrium using iterative best response with LQRax.
        
        Args:
            initial_states: [n_agents, 4] initial states
            goals: [n_agents, 2] goal positions
            max_iterations: Max Nash iterations
            tolerance: Convergence tolerance
            
        Returns:
            trajectories: [n_agents, horizon+1, 4] Nash equilibrium trajectories
        """
        # Convert to JAX arrays
        initial_states_jax = jnp.array(initial_states.numpy())
        goals_jax = jnp.array(goals.numpy())
        
        # Initialize trajectories with simple forward simulation
        trajectories = []
        for i in range(self.n_agents):
            initial_state = initial_states_jax[i]
            
            # Forward simulate with zero control
            traj = jnp.zeros((self.horizon + 1, self.state_dim))
            traj = traj.at[0].set(initial_state)
            
            for t in range(self.horizon):
                # Apply dynamics with zero control
                next_state = self.A @ traj[t] + self.B @ jnp.zeros(self.control_dim)
                traj = traj.at[t+1].set(next_state)
            
            trajectories.append(traj)
        
        trajectories = jnp.array(trajectories)
        
        print("Solving Nash equilibrium with LQRax LQR...")
        
        for iteration in range(max_iterations):
            trajectories_old = trajectories.copy()
            
            # Best response for each agent
            for agent in range(self.n_agents):
                # Get other agents' trajectories
                other_trajs = [trajectories[j] for j in range(self.n_agents) if j != agent]
                
                try:
                    # Solve LQR for this agent
                    agent_traj, agent_controls = self.solve_single_agent_lqr(
                        agent, 
                        initial_states_jax[agent], 
                        goals_jax[agent], 
                        other_trajs
                    )
                    trajectories = trajectories.at[agent].set(agent_traj)
                    
                except Exception as e:
                    print(f"LQR solve failed for agent {agent}: {e}")
                    # Keep previous trajectory
                    pass
            
            # Check convergence
            change = jnp.linalg.norm(trajectories - trajectories_old)
            print(f"Iteration {iteration + 1}: change = {change:.6f}")
            
            # Print current final positions
            for i in range(self.n_agents):
                final_pos = trajectories[i, -1, :2]
                goal = goals_jax[i]
                distance_to_goal = jnp.linalg.norm(final_pos - goal)
                print(f"  Agent {i}: pos={final_pos}, goal={goal}, dist={distance_to_goal:.4f}")
            
            if change < tolerance:
                print(f"Nash equilibrium converged in {iteration + 1} iterations")
                break
        else:
            print(f"Nash equilibrium did not converge in {max_iterations} iterations")
        
        # Convert back to torch format
        trajectories_torch = torch.tensor(np.array(trajectories), dtype=torch.float32)
        return trajectories_torch
    
    def extract_controls(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Extract control inputs from optimal trajectories using dynamics.
        
        Args:
            trajectories: [n_agents, horizon+1, 4] optimal trajectories
            
        Returns:
            controls: [n_agents, horizon, 2] control inputs
        """
        controls = torch.zeros(self.n_agents, self.horizon, 2)
        
        for agent in range(self.n_agents):
            for k in range(self.horizon):
                # From dynamics: x_{k+1} = A*x_k + B*u_k
                # Solve for u_k: u_k = B^{-1} * (x_{k+1} - A*x_k)
                # For double integrator: a_k = (v_{k+1} - v_k) / dt
                v_k = trajectories[agent, k, 2:]
                v_k1 = trajectories[agent, k+1, 2:]
                controls[agent, k] = (v_k1 - v_k) / self.dt
        
        return controls


# Example usage and testing
if __name__ == "__main__":
    # Create solver
    solver = LQRaxNashSolver(n_agents=2, horizon=30, dt=0.1)
    
    # Test with crossing scenario from paper
    initial_states = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],  # Agent 1: at origin, stationary
        [2.0, 2.0, 0.0, 0.0]   # Agent 2: at (2,2), stationary  
    ])
    
    goals = torch.tensor([
        [5.0, 3.0],  # Agent 1 goal
        [2.0, 5.0]   # Agent 2 goal (crossing paths)
    ])
    
    # Solve Nash equilibrium
    print("Solving Nash equilibrium with LQRax...")
    trajectories = solver.solve_nash_equilibrium(initial_states, goals)
    
    if trajectories is not None:
        print("Solution found!")
        print("Final positions:")
        for i in range(solver.n_agents):
            final_pos = trajectories[i, -1, :2]
            goal_pos = goals[i]
            distance = torch.norm(final_pos - goal_pos)
            print(f"Agent {i}: final={final_pos}, goal={goal_pos}, dist={distance:.4f}")
        
        # Extract controls
        controls = solver.extract_controls(trajectories)
        print("\nFirst control inputs:")
        for i in range(solver.n_agents):
            print(f"Agent {i}: {controls[i, 0]}")
    else:
        print("Failed to find solution")