import jax
import jax.numpy as jnp
from typing import Tuple

from load_config import load_config

# Load configuration
config = load_config()
T_total = config.game.T_total
dt = config.game.dt
Q = jnp.diag(jnp.array(config.optimization.Q))  # State cost weights [x, y, vx, vy]
R = jnp.diag(jnp.array(config.optimization.R))  # Control cost weights [ax, ay]


def solve_masked_game_differentiable(agents: list, initial_states: list, target_positions: jnp.ndarray,
                                   mask_values: jnp.ndarray = None, num_iters: int = 10, 
                                   reference_trajectories: list = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve the masked game using a fully differentiable approach.
    
    This version ensures proper gradient flow through the entire game-solving pipeline
    by using JAX-compatible operations throughout.
    
    Args:
        agents: List of selected agents (used for configuration only)
        initial_states: List of initial states for selected agents
        target_positions: Target positions for selected agents
        mask_values: Mask values for collision avoidance
        num_iters: Number of optimization iterations
        
    Returns:
        Tuple of (state_trajectories, control_trajectories)
    """
    n_selected = len(agents)
    
    # Convert lists to JAX arrays for better performance and differentiability
    initial_states_array = jnp.stack([jnp.array(s) for s in initial_states])  # (n_selected, 4)
    
    # Create reference trajectories (EXACTLY like reference generation)
    # Linear interpolation from initial position to target position
    reference_trajectories = []
    for i in range(n_selected):
        start_pos = initial_states[i][:2]  # Extract x, y position
        target_pos = target_positions[i]
        # Linear interpolation over time steps (EXACTLY like reference generation)
        ref_traj = jnp.linspace(start_pos, target_pos, T_total)
        # Convert to full state trajectory [x, y, vx, vy]
        ref_states = jnp.zeros((T_total, 4))
        for t in range(T_total):
            pos = ref_traj[t]
            vel = (target_pos - start_pos) / (T_total * dt)  # Constant velocity
            ref_states = ref_states.at[t, :].set(jnp.concatenate([pos, vel]))
        reference_trajectories.append(ref_states)
    
    # Create goal trajectories for compatibility (not used in cost function anymore)
    goal_trajectories = jnp.stack([
        jnp.tile(target_positions[i], (T_total, 1))
        for i in range(n_selected)
    ])  # (n_selected, T_total, 2)
    
    # Initialize control trajectories
    initial_controls = jnp.zeros((n_selected, T_total, 2))
    
    # Optimization parameters (following ilqgames_example.py pattern)
    step_size = config.optimization.step_size  # Conservative step size similar to original
    
    # Use reasonable number of iterations for convergence
    # training_iters = min(num_iters, config.optimization.num_iters)  # Reasonable iterations for stable convergence
    training_iters = config.optimization.num_iters
    
    def dynamics_function(x, u):
        """Point mass dynamics: [x, y, vx, vy] with controls [ax, ay]"""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy  
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])
    
    def integrate_dynamics(x0, u_traj):
        """Integrate dynamics forward using Euler integration"""
        def step_fn(x, u):
            x_next = x + dt * dynamics_function(x, u)
            return x_next, x_next
        
        _, x_traj = jax.lax.scan(step_fn, x0, u_traj)
        return x_traj
    
    def linearize_dynamics_at_trajectory(x0, u_traj):
        """Linearize dynamics around a trajectory"""
        x_traj = integrate_dynamics(x0, u_traj)
        
        # Compute Jacobians A = df/dx and B = df/du
        def compute_jacobians(x, u):
            A = jax.jacfwd(dynamics_function, argnums=0)(x, u)
            B = jax.jacfwd(dynamics_function, argnums=1)(x, u)
            return A, B
        
        A_traj, B_traj = jax.vmap(compute_jacobians)(x_traj, u_traj)
        return x_traj, A_traj, B_traj
    
    def compute_cost_gradients(agent_idx, x_traj, u_traj, goal_traj, all_x_trajs, mask_values, ref_traj=None):
        """Compute cost gradients for a single agent"""
        def single_step_cost(x, u, goal_x, other_xs, ref_x=None):
            # Navigation cost - track reference trajectory (EXACTLY like reference generation)
            nav_cost = jnp.sum(jnp.square(x[:2] - ref_x[:2]))
            
            # Collision avoidance costs - exponential penalty for proximity to other agents
            # (EXACTLY like reference generation) with proper masking
            collision_cost = 0.0
            
            # Compute collision cost with each other agent
            if other_xs.shape[0] > 0:
                # other_xs contains the states of other agents at this timestep
                # For agent_idx = 0 (ego), other agents are at indices [1, 2, ..., n-1] in the full game
                # For agent_idx = i, other agents are at indices [0, 1, ..., i-1, i+1, ..., n-1]
                
                # Compute collision cost for each other agent
                for i, other_x in enumerate(other_xs):
                    distance_squared = jnp.sum(jnp.square(x[:2] - other_x[:2]))
                    
                    if agent_idx == 0:  # Ego agent
                        # For ego agent, collision cost is weighted by the mask values of other agents
                        # other_xs[i] corresponds to agent i+1, use mask index i
                        mask_idx = i
                        if mask_idx < len(mask_values):
                            agent_mask_value = mask_values[mask_idx]
                            collision_cost += config.optimization.collision_weight * agent_mask_value * jnp.exp(-config.optimization.collision_scale * distance_squared)
                    else:  # Non-ego agent
                        # For other agents, collision cost is always full (they're always "selected" when in the game)
                        collision_cost += config.optimization.collision_weight * jnp.exp(-config.optimization.collision_scale * distance_squared)
            
            # Control cost - simplified without velocity scaling (EXACTLY like reference generation)
            ctrl_cost = config.optimization.control_weight * jnp.sum(jnp.square(u))
            
            return nav_cost + collision_cost + ctrl_cost
        
        # Get other agents' trajectories (excluding current agent)
        other_indices = jnp.array([i for i in range(n_selected) if i != agent_idx])
        
        if len(other_indices) > 0:
            other_x_trajs = all_x_trajs[other_indices]
        else:
            other_x_trajs = jnp.zeros((0, T_total, 4))
        
        # Compute gradients w.r.t. state and control
        if len(other_indices) > 0:
            other_x_transposed = other_x_trajs.transpose(1, 0, 2)
        else:
            other_x_transposed = jnp.zeros((T_total, 0, 4))
            
        # Use the reference trajectory for this agent (created from goals)
        ref_traj_array = reference_trajectories[agent_idx]
        
        a_traj = jax.vmap(jax.grad(single_step_cost, argnums=0))(
            x_traj, u_traj, goal_traj, other_x_transposed, ref_traj_array)
        b_traj = jax.vmap(jax.grad(single_step_cost, argnums=1))(
            x_traj, u_traj, goal_traj, other_x_transposed, ref_traj_array)
        
        return a_traj, b_traj
    
    def solve_lqr_subproblem(A_traj, B_traj, a_traj, b_traj, agent):
        """Solve LQR subproblem using proper iLQR solve method like ilqgames_example.py"""
        # Use the agent's built-in solve method which implements the Riccati equation
        v_traj, z_traj = agent.solve(A_traj, B_traj, a_traj, b_traj)
        return v_traj
    
    def optimization_step(carry, _):
        """Single optimization step - fully differentiable"""
        control_trajectories = carry
        
        # Step 1: Linearize dynamics for all agents
        x_trajs = []
        A_trajs = []
        B_trajs = []
        
        for i in range(n_selected):
            x_traj, A_traj, B_traj = linearize_dynamics_at_trajectory(
                initial_states_array[i], control_trajectories[i])
            x_trajs.append(x_traj)
            A_trajs.append(A_traj)
            B_trajs.append(B_traj)
        
        x_trajs = jnp.stack(x_trajs)  # (n_selected, T_total, 4)
        A_trajs = jnp.stack(A_trajs)  # (n_selected, T_total, 4, 4)
        B_trajs = jnp.stack(B_trajs)  # (n_selected, T_total, 4, 2)
        
        # Step 2: Compute cost gradients for all agents
        a_trajs = []
        b_trajs = []
        
        for i in range(n_selected):
            a_traj, b_traj = compute_cost_gradients(
                i, x_trajs[i], control_trajectories[i], 
                goal_trajectories[i], x_trajs, mask_values, None)  # ref_traj not used anymore
            a_trajs.append(a_traj)
            b_trajs.append(b_traj)
        
        a_trajs = jnp.stack(a_trajs)  # (n_selected, T_total, 4)
        b_trajs = jnp.stack(b_trajs)  # (n_selected, T_total, 2)
        
        # Step 3: Solve LQR subproblems for all agents (like ilqgames_example.py)
        control_updates = []
        for i in range(n_selected):
            v_traj = solve_lqr_subproblem(A_trajs[i], B_trajs[i], a_trajs[i], b_trajs[i], agents[i])
            control_updates.append(v_traj)
        
        control_updates = jnp.stack(control_updates)  # (n_selected, T_total, 2)
        
        # Step 4: Update control trajectories
        new_control_trajectories = control_trajectories + step_size * control_updates
        
        return new_control_trajectories, x_trajs
    
    # Use JAX scan for differentiable optimization
    final_carry, scan_outputs = jax.lax.scan(
        optimization_step, initial_controls, None, length=training_iters)
    
    # Get final results
    final_control_trajectories = final_carry  # Final control trajectories
    final_state_trajectories = scan_outputs[-1]  # Last state trajectories
    
    # Convert back to list format for compatibility
    final_state_list = [final_state_trajectories[i] for i in range(n_selected)]
    final_control_list = [final_control_trajectories[i] for i in range(n_selected)]
    
    return final_state_list, final_control_list