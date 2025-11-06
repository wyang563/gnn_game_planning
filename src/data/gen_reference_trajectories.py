import json
import jax.numpy as jnp
from pathlib import Path
from tqdm import tqdm

# Import from the main lqrax module
import sys
import os
# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from load_config import load_config, setup_jax_config, get_device_config
from solver.solve import create_agent_setup, create_loss_functions, solve_ilqgames_parallel_no_mask, save_trajectory_sample, plot_sample_trajectories
import random

def generate_reference_trajectories(**kwargs):
    # extract inputs
    n_agents = kwargs["n_agents"]
    tsteps = kwargs["tsteps"]
    dt = kwargs["dt"]
    num_iters = kwargs["num_iters"]
    step_size = kwargs["step_size"]
    Q = kwargs["Q"]
    R = kwargs["R"]
    boundary_size = kwargs["boundary_size"]
    init_type = kwargs["init_type"]
    device = kwargs["device"]
    output_dir = kwargs["output_dir"]
    num_samples = kwargs["num_samples"]
    gen_type = kwargs["gen_type"]
    weights = kwargs["weights"]
    x_dim = kwargs["x_dim"]
    u_dim = kwargs["u_dim"]
    dt = kwargs["dt"]

    num_existing_samples = len(os.listdir(output_dir)) // 2
    start_id = num_existing_samples
    upper_bound_agents = n_agents

    print("Starting from sample ID: ", start_id)

    for sample_id in tqdm(range(start_id, num_samples), total=num_samples - start_id, desc="Generating reference trajectories"):
        # create agent setup
        if gen_type == "fixed":
            agents, initial_states, reference_trajectories, target_positions = create_agent_setup(n_agents, init_type, x_dim, u_dim, dt, Q, R, tsteps, boundary_size, device, weights)
            create_loss_functions(agents, "no_mask")
        elif gen_type == "variable":
            # make it twice as more likely to generate n > 5 agents than n < 5 agents
            selection_pool = []
            for i in range(2, upper_bound_agents + 1):
                if i < 4:
                    selection_pool.extend([i] * 1)
                elif 4 <= i < 6:
                    selection_pool.extend([i] * 2)
                else:
                    selection_pool.extend([i] * 3)
            n_agents = random.choice(selection_pool)
            # recalibrate boundary size based on number of agents
            boundary_size = n_agents**(0.5)  * 1.75
            agents, initial_states, reference_trajectories, target_positions = create_agent_setup(n_agents, init_type, x_dim, u_dim, dt, Q, R, tsteps, boundary_size, device, weights)
            create_loss_functions(agents, "no_mask")
        else:
            raise ValueError(f"Invalid generation type: {gen_type}")

        # solve iLQGames
        state_trajs, control_trajs, _ = solve_ilqgames_parallel_no_mask(agents, initial_states, reference_trajectories, num_iters, u_dim, tsteps, step_size, device)

        # save trajectory sample
        sample_data = save_trajectory_sample(
            sample_id, 
            n_agents, 
            tsteps, 
            dt, 
            jnp.array([initial_states[i][:2] for i in range(n_agents)]),  # Extract positions
            jnp.array([target_positions[i][:2] for i in range(n_agents)]),  # Extract positions
            state_trajs,
            control_trajs
        )

        json_filename = f"ref_traj_sample_{sample_id:03d}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(sample_data, f, indent=2)

        # create and save trajectory plot
        plot_path = os.path.join(output_dir, f"ref_traj_sample_{sample_id:03d}.png")
        plot_sample_trajectories(n_agents, sample_data, boundary_size, plot_path)

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

    # create output directory
    gen_type = config.reference_generation.reference_gen_type
    if gen_type == "fixed":
        output_dir = Path(config.reference_generation.save_dir_fixed)
    elif gen_type == "variable":
        output_dir = Path(config.reference_generation.save_dir_variable)
    else:
        raise ValueError(f"Invalid generation type: {gen_type}")
    
    output_dir = os.path.join("src/data", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_samples = config.reference_generation.num_samples

    # generate reference trajectories
    args = {
        "n_agents": n_agents,
        "tsteps": tsteps,
        "dt": dt,
        "num_iters": num_iters,
        "step_size": step_size,
        "Q": Q,
        "R": R,
        "boundary_size": boundary_size,
        "init_type": init_type,
        "device": device,
        "output_dir": output_dir,
        "num_samples": num_samples,
        "gen_type": gen_type,
        "weights": weights,
        "x_dim": x_dim,
        "u_dim": u_dim,
        "dt": dt,
    }
    generate_reference_trajectories(**args)