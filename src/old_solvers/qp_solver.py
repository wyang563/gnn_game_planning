import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from qpth.qp import QPFunction
from typing import Tuple, Optional


class QPNashSolver:
    """
    Quadratic Programming solver for multi-agent Nash equilibrium using qpth.
    
    Solves a linear-quadratic game with:
    - Double integrator dynamics
    - Position, velocity, and control bounds
    - Goal tracking and collision avoidance objectives
    """
    
    def __init__(self, 
                 n_agents: int = 2, 
                 horizon: int = 50,
                 dt: float = 0.1,
                 w1: float = 1.0,  # goal tracking weight (default from reference)
                 w2: float = 1.0,  # velocity penalty weight (default from reference)
                 w3: float = 0.1,  # control effort weight (default from reference)
                 w4: float = 1.0,  # collision avoidance weight (default from reference)
                 eps: float = 1e-4):
        """
        Initialize QP Nash solver.
        
        Args:
            n_agents: Number of agents
            horizon: Planning horizon T
            dt: Time step Delta
            w1-w4: Cost function weights
            eps: Regularization term for positive definiteness
        """
        self.n_agents = n_agents
        self.horizon = horizon
        self.dt = dt
        self.w1, self.w2, self.w3, self.w4 = w1, w2, w3, w4
        self.eps = eps
        
        # State and control dimensions
        self.state_dim = 4  # [px, py, vx, vy]
        self.control_dim = 2  # [ax, ay]
        
        # Total optimization variable dimensions per agent
        self.n_states = self.state_dim * (horizon + 1)  # x_0, ..., x_T
        self.n_controls = self.control_dim * horizon     # u_0, ..., u_{T-1}
        self.n_vars_per_agent = self.n_states + self.n_controls
        self.n_vars_total = self.n_vars_per_agent * n_agents
        
        # Setup dynamics matrices
        self.A = torch.zeros(4, 4)
        self.A[:2, :2] = torch.eye(2)  # position to position
        self.A[:2, 2:] = torch.eye(2) * self.dt  # velocity to position  
        self.A[2:, 2:] = torch.eye(2)  # velocity to velocity
        
        # B = [0; I_2*dt]
        self.B = torch.zeros(4, 2)
        self.B[2:, :] = torch.eye(2) * self.dt
        
        # Default bounds (can be overridden)
        self.pos_bounds = (-10.0, 10.0)
        self.vel_bounds = (-5.0, 5.0) 
        self.accel_bounds = (-2.0, 2.0)
        
    def _build_dynamics_constraints(self, initial_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build equality constraints for dynamics: x_{k+1} = A*x_k + B*u_k
        Also includes initial state constraints: x_0 = x_init
        
        Args:
            initial_states: Initial states [n_agents, 4]
            
        Returns:
            A_eq: Equality constraint matrix
            b_eq: Equality constraint vector
        """
        # Constraints: initial states + dynamics for each timestep
        n_initial_constraints = self.n_agents * self.state_dim
        n_dynamics_constraints = self.n_agents * self.horizon * self.state_dim
        n_eq_constraints = n_initial_constraints + n_dynamics_constraints
        
        A_eq = torch.zeros(n_eq_constraints, self.n_vars_total)
        b_eq = torch.zeros(n_eq_constraints)
        
        constraint_idx = 0
        
        # Initial state constraints: x_0 = x_init
        for agent in range(self.n_agents):
            agent_offset = agent * self.n_vars_per_agent
            for state_idx in range(self.state_dim):
                row = constraint_idx + state_idx
                A_eq[row, agent_offset + state_idx] = 1.0
                b_eq[row] = initial_states[agent, state_idx]
            constraint_idx += self.state_dim
        
        # Dynamics constraints: x_{k+1} - A*x_k - B*u_k = 0
        for agent in range(self.n_agents):
            agent_offset = agent * self.n_vars_per_agent
            
            for k in range(self.horizon):
                # Indices for states and controls
                x_k_start = agent_offset + k * self.state_dim
                x_k1_start = agent_offset + (k + 1) * self.state_dim
                u_k_start = agent_offset + self.n_states + k * self.control_dim
                
                # x_{k+1} - A*x_k - B*u_k = 0
                for state_idx in range(self.state_dim):
                    row = constraint_idx + state_idx
                    
                    # +x_{k+1}
                    A_eq[row, x_k1_start + state_idx] = 1.0
                    
                    # -A*x_k  
                    for j in range(self.state_dim):
                        A_eq[row, x_k_start + j] = -self.A[state_idx, j]
                    
                    # -B*u_k
                    for j in range(self.control_dim):
                        A_eq[row, u_k_start + j] = -self.B[state_idx, j]
                
                constraint_idx += self.state_dim
        
        return A_eq, b_eq
    
    def _build_bound_constraints(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build inequality constraints for bounds on states and controls.
        
        Returns:
            G: Inequality constraint matrix (G*z <= h)
            h: Inequality constraint vector
        """
        # Count constraints: 2 bounds × (pos + vel + control) × horizon × agents
        n_pos_constraints = 4 * (self.horizon + 1) * self.n_agents  # px_min, px_max, py_min, py_max
        n_vel_constraints = 4 * (self.horizon + 1) * self.n_agents  # vx_min, vx_max, vy_min, vy_max  
        n_ctrl_constraints = 4 * self.horizon * self.n_agents       # ax_min, ax_max, ay_min, ay_max
        
        n_ineq_constraints = n_pos_constraints + n_vel_constraints + n_ctrl_constraints
        G = torch.zeros(n_ineq_constraints, self.n_vars_total)
        h = torch.zeros(n_ineq_constraints)
        
        constraint_idx = 0
        
        for agent in range(self.n_agents):
            agent_offset = agent * self.n_vars_per_agent
            
            # Position and velocity bounds for all timesteps
            for k in range(self.horizon + 1):
                state_start = agent_offset + k * self.state_dim
                
                # Position bounds: px_min <= px <= px_max, py_min <= py <= py_max
                for dim in range(2):  # x, y dimensions
                    pos_idx = state_start + dim
                    
                    # Lower bound: -px <= -px_min  =>  px >= px_min
                    G[constraint_idx, pos_idx] = -1.0
                    h[constraint_idx] = -self.pos_bounds[0]
                    constraint_idx += 1
                    
                    # Upper bound: px <= px_max
                    G[constraint_idx, pos_idx] = 1.0  
                    h[constraint_idx] = self.pos_bounds[1]
                    constraint_idx += 1
                
                # Velocity bounds: vx_min <= vx <= vx_max, vy_min <= vy <= vy_max
                for dim in range(2):  # x, y dimensions
                    vel_idx = state_start + 2 + dim
                    
                    # Lower bound: -vx <= -vx_min  =>  vx >= vx_min
                    G[constraint_idx, vel_idx] = -1.0
                    h[constraint_idx] = -self.vel_bounds[0]
                    constraint_idx += 1
                    
                    # Upper bound: vx <= vx_max
                    G[constraint_idx, vel_idx] = 1.0
                    h[constraint_idx] = self.vel_bounds[1]
                    constraint_idx += 1
            
            # Control bounds for all timesteps
            for k in range(self.horizon):
                control_start = agent_offset + self.n_states + k * self.control_dim
                
                # Control bounds: ax_min <= ax <= ax_max, ay_min <= ay <= ay_max
                for dim in range(2):  # x, y dimensions
                    ctrl_idx = control_start + dim
                    
                    # Lower bound: -ax <= -ax_min  =>  ax >= ax_min
                    G[constraint_idx, ctrl_idx] = -1.0
                    h[constraint_idx] = -self.accel_bounds[0]
                    constraint_idx += 1
                    
                    # Upper bound: ax <= ax_max
                    G[constraint_idx, ctrl_idx] = 1.0
                    h[constraint_idx] = self.accel_bounds[1]
                    constraint_idx += 1
        
        return G, h
    
    def _build_objective_matrix(self, goals: torch.Tensor) -> torch.Tensor:
        """
        Build quadratic objective matrix Q for agent i.
        
        Objective: sum_k [w1*||p_k - p_goal||^2 + w2*||v_k||^2 + w3*||u_k||^2 + w4*collision_term]
        
        Args:
            goals: Target positions [n_agents, 2]
            
        Returns:
            Q: Quadratic cost matrix [n_vars_total, n_vars_total]
        """
        Q = torch.zeros(self.n_vars_total, self.n_vars_total)
        
        for agent in range(self.n_agents):
            agent_offset = agent * self.n_vars_per_agent
            goal = goals[agent]
            
            # Add costs for each timestep
            for k in range(self.horizon + 1):
                state_start = agent_offset + k * self.state_dim
                
                # Goal tracking cost: w1 * ||p_k - p_goal||^2
                for dim in range(2):  # position dimensions
                    pos_idx = state_start + dim
                    # qpth uses 0.5 * z^T Q z + p^T z, so set Q = 2*w1*I
                    Q[pos_idx, pos_idx] += 2.0 * self.w1
                
                # Velocity penalty: w2 * ||v_k||^2  
                for dim in range(2):  # velocity dimensions
                    vel_idx = state_start + 2 + dim
                    # Set Q = 2*w2*I
                    Q[vel_idx, vel_idx] += 2.0 * self.w2
            
            # Control effort cost: w3 * ||u_k||^2
            for k in range(self.horizon):
                control_start = agent_offset + self.n_states + k * self.control_dim
                for dim in range(2):  # control dimensions
                    ctrl_idx = control_start + dim
                    # Set Q = 2*w3*I
                    Q[ctrl_idx, ctrl_idx] += 2.0 * self.w3
        
        # Add collision terms (coupling between agents)
        for k in range(self.horizon + 1):
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    i_offset = i * self.n_vars_per_agent + k * self.state_dim
                    j_offset = j * self.n_vars_per_agent + k * self.state_dim
                    
                    # Quadratic term for distance: (p_i - p_j)^T (p_i - p_j)
                    # Repulsive cost uses negative sign on this quantity.
                    for dim in range(2):  # Only position dimensions
                        # p_i^T*p_i and p_j^T*p_j contributions with negative sign
                        Q[i_offset + dim, i_offset + dim] += 2.0 * self.w4
                        Q[j_offset + dim, j_offset + dim] += 2.0 * self.w4
                        # Cross terms switch sign accordingly: +4*w4
                        Q[i_offset + dim, j_offset + dim] -= 4.0 * self.w4
                        Q[j_offset + dim, i_offset + dim] -= 4.0 * self.w4
        
        # Add regularization for positive definiteness
        Q += self.eps * torch.eye(self.n_vars_total)
        
        return Q
    
    def _build_linear_term(self, goals: torch.Tensor) -> torch.Tensor:
        """
        Build linear term p for the objective.
        
        Args:
            goals: Target positions [n_agents, 2]
            
        Returns:
            p: Linear cost vector [n_vars_total]
        """
        p = torch.zeros(self.n_vars_total)
        
        for agent in range(self.n_agents):
            agent_offset = agent * self.n_vars_per_agent
            goal = goals[agent]
            
            # Linear terms from goal tracking: -2*w1*p_goal^T*p_k
            for k in range(self.horizon + 1):
                state_start = agent_offset + k * self.state_dim
                for dim in range(2):
                    pos_idx = state_start + dim
                    # With Q scaled by 2*w1, p should be -2*w1*goal
                    p[pos_idx] = -2.0 * self.w1 * goal[dim]
        
        return p
    
    def solve_single_agent_qp(self, 
                            agent_idx: int,
                            initial_states: torch.Tensor,
                            goals: torch.Tensor,
                            other_trajectories: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Solve QP for a single agent given fixed trajectories of others.
        
        Args:
            agent_idx: Index of agent to optimize
            initial_states: Initial states [n_agents, 4]
            goals: Goal positions [n_agents, 2] 
            other_trajectories: Fixed trajectories of other agents [n_other_agents, horizon+1, 4]
            
        Returns:
            Optimal trajectory for the agent [horizon+1, 4] or None if infeasible
        """
        try:
            # Build QP components
            Q = self._build_objective_matrix(goals)
            p = self._build_linear_term(goals)
            A_eq, b_eq = self._build_dynamics_constraints(initial_states)
            G, h = self._build_bound_constraints()
            
            # Note: Original paper uses only soft goal penalties, no hard terminal constraints
            
            # Convert to Variables for qpth
            Q_var = Variable(Q.unsqueeze(0))  # Add batch dimension
            p_var = Variable(p.unsqueeze(0))  # Add batch dimension
            G_var = Variable(G.unsqueeze(0))  # Add batch dimension  
            h_var = Variable(h.unsqueeze(0))  # Add batch dimension
            A_var = Variable(A_eq.unsqueeze(0))  # Add batch dimension
            b_var = Variable(b_eq.unsqueeze(0))  # Add batch dimension
            
            # Solve QP using qpth
            solution = QPFunction(verbose=False)(Q_var, p_var, G_var, h_var, A_var, b_var)

            # Optional: check equality residuals for debugging numerical issues
            try:
                z = solution[0]
                eq_resid = torch.norm(A_eq @ z - b_eq)
                # Uncomment for deeper debugging
                # print(f"Equality residual norm: {eq_resid:.3e}")
            except Exception:
                pass
            
            if solution is not None:
                # Extract agent's trajectory
                agent_offset = agent_idx * self.n_vars_per_agent
                agent_states = solution[0, agent_offset:agent_offset + self.n_states]
                agent_trajectory = agent_states.view(self.horizon + 1, self.state_dim)
                return agent_trajectory
            else:
                return None
                
        except Exception as e:
            print(f"QP solve failed for agent {agent_idx}: {e}")
            return None
    
    def solve_nash_equilibrium(self, 
                             initial_states: torch.Tensor,
                             goals: torch.Tensor,
                             max_iterations: int = 50,
                             tolerance: float = 1e-3) -> Optional[torch.Tensor]:
        """
        Solve Nash equilibrium using iterative best response.
        
        Args:
            initial_states: Initial states [n_agents, 4]
            goals: Goal positions [n_agents, 2]
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Nash equilibrium trajectories [n_agents, horizon+1, 4] or None if failed
        """
        # Initialize with random trajectories
        trajectories = torch.randn(self.n_agents, self.horizon + 1, self.state_dim)
        
        # Set initial states
        trajectories[:, 0, :] = initial_states
        
        for iteration in range(max_iterations):
            trajectories_old = trajectories.clone()
            
            # Best response for each agent
            for agent in range(self.n_agents):
                # Get trajectories of other agents
                other_trajectories = torch.cat([
                    trajectories[:agent], 
                    trajectories[agent+1:]
                ], dim=0)
                
                # Solve QP for this agent
                agent_trajectory = self.solve_single_agent_qp(
                    agent, initial_states, goals, other_trajectories
                )
                
                if agent_trajectory is not None:
                    trajectories[agent] = agent_trajectory
                else:
                    print(f"Failed to solve QP for agent {agent} at iteration {iteration}")
                    return None
            
            # Check convergence
            change = torch.norm(trajectories - trajectories_old)
            print(f"Iteration {iteration + 1}: change = {change:.6f}")
            
            # Print current final positions
            for i in range(self.n_agents):
                final_pos = trajectories[i, -1, :2]
                goal = goals[i]
                distance_to_goal = torch.norm(final_pos - goal)
                print(f"  Agent {i}: pos={final_pos}, goal={goal}, dist={distance_to_goal:.4f}")
            
            if change < tolerance:
                print(f"Nash equilibrium converged in {iteration + 1} iterations")
                return trajectories
        
        print(f"Nash equilibrium did not converge in {max_iterations} iterations")
        return trajectories  # Return best effort
    
    def extract_controls(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Extract control inputs from optimal trajectories.
        
        Args:
            trajectories: Optimal trajectories [n_agents, horizon+1, 4]
            
        Returns:
            Control inputs [n_agents, horizon, 2]
        """
        controls = torch.zeros(self.n_agents, self.horizon, self.control_dim)
        
        for agent in range(self.n_agents):
            for k in range(self.horizon):
                # u_k = (x_{k+1} - A*x_k) / B (approximately)
                # For double integrator: a_k = (v_{k+1} - v_k) / dt
                v_k = trajectories[agent, k, 2:]
                v_k1 = trajectories[agent, k+1, 2:]
                controls[agent, k] = (v_k1 - v_k) / self.dt
        
        return controls


# Example usage and testing
if __name__ == "__main__":
    # Create solver
    solver = QPNashSolver(n_agents=2, horizon=50, dt=0.1)
    
    # (position) concatened with inital velocity
    initial_states = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],  
        [2.0, 2.0, 0.0, 0.0]   
    ])
    
    goals = torch.tensor([
        [5.0, 3.0],  
        [2.0, 5.0]   
    ])
    
    # Solve Nash equilibrium
    print("Solving Nash equilibrium...")
    trajectories = solver.solve_nash_equilibrium(initial_states, goals)
    
    if trajectories is not None:
        print("Solution found!")
        print("Final positions:")
        for i in range(solver.n_agents):
            final_pos = trajectories[i, :, :2]
            print(f"Agent {i}: {final_pos}")
        
        # Extract controls
        controls = solver.extract_controls(trajectories)
        print("\nFirst control inputs:")
        for i in range(solver.n_agents):
            print(f"Agent {i}: {controls[i, 0]}")
    else:
        print("Failed to find solution")
