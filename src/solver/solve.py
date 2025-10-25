import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt
from jax import vmap, jit, grad

# Import from the main lqrax module
import sys
import os
# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from solver.point_agent import PointAgent
from load_config import load_config, setup_jax_config, get_device_config, ConfigLoader
from utils.goal_init import origin_init_collision, random_init

def create_loss_functions(agents, mode):
    for agent in agents:
        if mode == "mask":
            agent.create_loss_functions_mask()
        elif mode == "no_mask":
            agent.create_loss_functions_no_mask()

# created batched loss functions (this is only for data generation when other x_trajs is the same size dims for all agents)
def create_batched_loss_functions_mask(agents, device):
    dummy_agent: PointAgent = agents[0]
    # Define batched functions that work on arrays of agent data
    def batched_linearize_dyn(x0s, u_trajs):
        """Batched version of linearize_dyn for all agents."""
        def single_agent_linearize_dyn(x0, u_traj):
            return dummy_agent.linearize_dyn(x0, u_traj)
        return vmap(single_agent_linearize_dyn)(x0s, u_trajs)

    def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
        """Batched version of linearize_loss for all agents."""
        def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask):
            return dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask)
        return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
    
    def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
        """Batched version of solve for all agents."""
        def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
            return dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
        return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
    
    def batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
        """Batched version of loss for all agents."""
        def single_agent_loss(x_traj, u_traj, ref_traj, other_traj, mask):
            return dummy_agent.loss(x_traj, u_traj, ref_traj, other_traj, mask)
        return vmap(single_agent_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)

    jit_batched_linearize_dyn = jit(batched_linearize_dyn, device=device)
    jit_batched_linearize_loss = jit(batched_linearize_loss, device=device)
    jit_batched_solve = jit(batched_solve, device=device)
    jit_batched_loss = jit(batched_loss, device=device)
    return jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss

def create_batched_loss_functions_no_mask(agents, device):
    dummy_agent: PointAgent = agents[0]
    # Define batched functions that work on arrays of agent data
    def batched_linearize_dyn(x0s, u_trajs):
        """Batched version of linearize_dyn for all agents."""
        def single_agent_linearize_dyn(x0, u_traj):
            return dummy_agent.linearize_dyn(x0, u_traj)
        return vmap(single_agent_linearize_dyn)(x0s, u_trajs)
    
    def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs):
        """Batched version of linearize_loss for all agents."""
        def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj):
            return dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj)
        return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs)
    
    def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
        """Batched version of solve for all agents."""
        def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
            return dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
        return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
    
    def batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs):
        """Batched version of loss for all agents."""
        def single_agent_loss(x_traj, u_traj, ref_traj, other_traj):
            return dummy_agent.loss(x_traj, u_traj, ref_traj, other_traj)
        return vmap(single_agent_loss)(x_trajs, u_trajs, ref_trajs, other_trajs)

    jit_batched_linearize_dyn = jit(batched_linearize_dyn, device=device)
    jit_batched_linearize_loss = jit(batched_linearize_loss, device=device)
    jit_batched_solve = jit(batched_solve, device=device)
    jit_batched_loss = jit(batched_loss, device=device)
    return jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss

def create_agent_setup(n_agents, setup_type, x_dim, u_dim, dt, Q, R, tsteps, init_position_range, device, weights):
    if setup_type == "random":
        init_positions, target_positions = random_init(n_agents, [-init_position_range, init_position_range])
    elif setup_type == "origin":
        init_positions, target_positions = origin_init_collision(n_agents, [-init_position_range, init_position_range])

    agents = []
    initial_states = []
    reference_trajectories = []
    collision_weight, collision_scale, ctrl_weight = weights

    for i in range(n_agents):
        agent = PointAgent(dt, x_dim=x_dim, u_dim=u_dim, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=ctrl_weight, device=device)
        agents.append(agent)
        initial_states.append(jnp.array([init_positions[i][0], init_positions[i][1], 0.0, 0.0])) # [x, y, vx, vy]

        start_pos = jnp.array(init_positions[i])
        target_pos = jnp.array(target_positions[i])
        ref_traj = jnp.linspace(start_pos, target_pos, tsteps)
        reference_trajectories.append(ref_traj)
    return agents, initial_states, reference_trajectories, target_positions

def solve_ilqgames_sequential(agents, initial_states, ref_trajs, num_iters, u_dim, tsteps, step_size):
    start = time.time()
    n_agents = len(agents)
    control_trajs = [jnp.zeros((tsteps, u_dim)) for _ in range(n_agents)]
    for _ in range(num_iters + 1):
        state_trajs = []
        A_trajs = []
        B_trajs = []
        for i in range(n_agents):
            x_traj, A_traj, B_traj = agents[i].compiled_linearize_dyn(initial_states[i], control_trajs[i])
            state_trajs.append(x_traj)
            A_trajs.append(A_traj)
            B_trajs.append(B_traj)

        a_trajs = []
        b_trajs = []
        for i in range(n_agents):
            other_states = jnp.array([state_trajs[j] for j in range(n_agents) if j != i])
            other_states = other_states.transpose(1, 0, 2)
            a_traj, b_traj = agents[i].compiled_linearize_loss(state_trajs[i], control_trajs[i], ref_trajs[i], other_states)
            a_trajs.append(a_traj)
            b_trajs.append(b_traj)
        control_updates = []

        for i in range(n_agents):
            v_traj, _ = agents[i].compiled_solve(A_trajs[i], B_trajs[i], a_trajs[i], b_trajs[i])
            control_updates.append(v_traj)
        for i in range(n_agents):
            control_trajs[i] += step_size * control_updates[i]
    end = time.time()
    total_time = end - start
    return state_trajs, control_trajs, total_time

def solve_ilqgames_parallel_mask(agents, initial_states, ref_trajs, num_iters, u_dim, tsteps, step_size, device, masks):
    start = time.time()
    n_agents = len(agents)

    # initialize batched functions
    jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss = create_batched_loss_functions_mask(agents, device)

    # initialize batched arrays
    control_trajs = jnp.zeros((n_agents, tsteps, u_dim))
    init_states = jnp.array([initial_states[i] for i in range(n_agents)])
    ref_trajs = jnp.array([ref_trajs[i] for i in range(n_agents)])

    for _ in range(num_iters + 1):
        x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(init_states, control_trajs)
        other_states = jnp.array([[x_trajs[j] for j in range(n_agents) if j != i] for i in range(n_agents)])
        other_states = other_states.transpose(0, 2, 1, 3)
        a_trajs, b_trajs = jit_batched_linearize_loss(x_trajs, control_trajs, ref_trajs, other_states, masks)
        v_trajs, _ = jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        control_trajs += step_size * v_trajs

    # decompose results into lists
    state_trajs = [x_trajs[i] for i in range(n_agents)]
    control_trajs = [control_trajs[i] for i in range(n_agents)]

    end = time.time()
    total_time = end - start
    return state_trajs, control_trajs, total_time

def solve_ilqgames_parallel_no_mask(agents, initial_states, ref_trajs, num_iters, u_dim, tsteps, step_size, device):
    start = time.time()
    n_agents = len(agents)

    # initialize batched functions
    jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss = create_batched_loss_functions_no_mask(agents, device)
    # initialize batched arrays
    control_trajs = jnp.zeros((n_agents, tsteps, u_dim))
    init_states = jnp.array([initial_states[i] for i in range(n_agents)])
    ref_trajs = jnp.array([ref_trajs[i] for i in range(n_agents)])

    for _ in range(num_iters + 1):
        x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(init_states, control_trajs)
        other_states = jnp.array([[x_trajs[j] for j in range(n_agents) if j != i] for i in range(n_agents)])
        other_states = other_states.transpose(0, 2, 1, 3)
        a_trajs, b_trajs = jit_batched_linearize_loss(x_trajs, control_trajs, ref_trajs, other_states)
        v_trajs, _ = jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        control_trajs += step_size * v_trajs

    # decompose results into lists
    state_trajs = [x_trajs[i] for i in range(n_agents)]
    control_trajs = [control_trajs[i] for i in range(n_agents)]

    end = time.time()
    total_time = end - start
    return state_trajs, control_trajs, total_time

def save_trajectory_sample(sample_id: int, n_agents: int, tsteps: int, dt: float, init_positions: jnp.ndarray, 
                          target_positions: jnp.ndarray, state_trajectories: List[jnp.ndarray],
                          control_trajectories: List[jnp.ndarray]) -> Dict[str, Any]:
    """
    Save a trajectory sample to a dictionary format.
    
    Args:
        sample_id: Sample identifier
        init_positions: Initial positions (n_agents, 2)
        target_positions: Target positions (n_agents, 2)
        state_trajectories: List of state trajectories for each agent
        control_trajectories: List of control trajectories for each agent
        
    Returns:
        sample_data: Dictionary containing the trajectory data
    """
    sample_data = {
        "sample_id": sample_id,
        "init_positions": init_positions.tolist(),  # Convert to list for JSON serialization
        "target_positions": target_positions.tolist(),  # Convert to list for JSON serialization
        "trajectories": {
            f"agent_{i}": {
                "states": state_trajectories[i].tolist(),  # (tsteps, state_dim)
                "controls": control_trajectories[i].tolist()  # (tsteps, control_dim)
            }
            for i in range(n_agents)
        },
        "metadata": {
            "n_agents": n_agents,
            "tsteps": tsteps,
            "dt": dt,
            "state_dim": 4,
            "control_dim": 2
        }
    }
    
    return sample_data

def plot_sample_trajectories(n_agents: int, sample_data: Dict[str, Any], boundary_size: float, save_path: str = None):
    """
    Plot trajectories for a single sample with proper boundary scaling.
    
    Args:
        sample_data: Sample data dictionary
        boundary_size: Size of the environment boundary for consistent scaling
        save_path: Path to save the plot (optional)
    """
    init_positions = np.array(sample_data["init_positions"])
    target_positions = np.array(sample_data["target_positions"])
    trajectories = sample_data["trajectories"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Square figure for equal aspect
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        states = np.array(trajectories[agent_key]["states"])
        
        # Extract positions
        positions = states[:, :2]  # (tsteps, 2)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 
               color=colors[i], linewidth=2, label=f'Agent {i}', alpha=0.8)
        
        # Plot start and end points
        ax.scatter(init_positions[i, 0], init_positions[i, 1], 
                  color=colors[i], s=120, marker='o', edgecolors='black', 
                  linewidth=2, label=f'Start {i}' if i == 0 else "")
        ax.scatter(target_positions[i, 0], target_positions[i, 1], 
                  color=colors[i], s=120, marker='*', edgecolors='black',
                  linewidth=2, label=f'Goal {i}' if i == 0 else "")
    
    # Set axis limits to show full boundary with small margin
    margin = 0.1
    ax.set_xlim(-boundary_size - margin, boundary_size + margin)
    ax.set_ylim(-boundary_size - margin, boundary_size + margin)
    
    # Draw boundary rectangle
    boundary_rect = plt.Rectangle((-boundary_size, -boundary_size), 
                                 2*boundary_size, 2*boundary_size,
                                 fill=False, edgecolor='gray', linewidth=2, 
                                 linestyle='--', alpha=0.5)
    ax.add_patch(boundary_rect)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Agent Trajectories (N={n_agents}, Boundary=±{boundary_size}m)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Plot saved to: {save_path}")
    
    # Don't show the plot during generation
    plt.close()

if __name__ == "__main__":
    config = load_config()
    setup_jax_config()
    device = get_device_config()
    print(f"Using device: {device}")

    # Extract parameters from configuration
    dt = config.game.dt
    tsteps = config.game.T_total
    n_agents = config.game.N_agents
    init_type = config.game.initiation_type
    boundary_size = config.get(f"game.boundary_size_{n_agents}p", 3.5)

    # Optimization parameters
    num_iters = config.optimization.num_iters
    step_size = config.optimization.step_size
    Q = jnp.diag(jnp.array(config.optimization.Q))
    R = jnp.diag(jnp.array(config.optimization.R))

    print(f"Configuration loaded:")
    print(f"  N agents: {n_agents}")
    print(f"  Time steps: {tsteps}, dt: {dt}")
    print(f"  Optimization: {num_iters} iters, step size: {step_size}")
    print(f"  Boundary size: {boundary_size}")

    x_dim = 4
    u_dim = 2
    weights = (config.optimization.collision_weight, config.optimization.collision_scale, config.optimization.control_weight)

    # create agent setup
    agents, initial_states, reference_trajectories, target_positions = create_agent_setup(n_agents, init_type, x_dim, u_dim, dt, Q, R, tsteps, boundary_size, device, weights)
    create_loss_functions(agents, "no_mask")
    # state_trajs, control_trajs, total_time = solve_ilqgames_sequential(agents, initial_states, reference_trajectories, num_iters, u_dim, tsteps, step_size)
    state_trajs, control_trajs, total_time = solve_ilqgames_parallel_no_mask(agents, initial_states, reference_trajectories, num_iters, u_dim, tsteps, step_size, device)
    print(f"Total solve time: {total_time}")

    sample_data = save_trajectory_sample(
        0, n_agents, tsteps, dt, 
        jnp.array([initial_states[i][:2] for i in range(n_agents)]),  # Extract positions
        target_positions, 
        state_trajs, 
        control_trajs 
    )

    save_path = "src/solver"
    plot_filename = f"test.png"
    plot_path = os.path.join(save_path, plot_filename)
    plot_sample_trajectories(sample_data, boundary_size, str(plot_path))


    