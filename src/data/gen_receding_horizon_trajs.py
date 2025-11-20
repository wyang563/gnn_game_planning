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
from solver.solve_by_horizon import solve_by_horizon
import glob
from solver.point_agent import PointAgent


def generate_receding_horizon_trajectories(**kwargs):
    # extract inputs
    n_agents = kwargs["n_agents"]
    tsteps = kwargs["tsteps"]
    dt = kwargs["dt"]
    num_iters = kwargs["num_iters"]
    step_size = kwargs["step_size"]
    Q = kwargs["Q"]
    R = kwargs["R"]
    boundary_size = kwargs["boundary_size"]
    device = kwargs["device"]
    out_dir = kwargs["output_dir"]
    gen_type = kwargs["gen_type"]
    weights = kwargs["weights"]
    x_dim = kwargs["x_dim"]
    u_dim = kwargs["u_dim"]
    dt = kwargs["dt"]
    planning_horizon = kwargs["planning_horizon"]
    mask_horizon = kwargs["mask_horizon"]
    mask_threshold = kwargs["mask_threshold"]
    model_type = "all"

    # get files from reference trajectory directory
    if gen_type == "fixed":
        reference_dir = os.path.join("src/data", f"reference_trajectories_{n_agents}p")
    elif gen_type == "variable":
        reference_dir = os.path.join("src/data", f"reference_trajectories_upto_{n_agents}p")
    else:
        raise ValueError(f"Invalid generation type: {gen_type}")

    json_files = sorted(glob.glob(os.path.join(reference_dir, "ref_traj_sample_*.json")))
    num_existing_samples = len(json_files)
    collision_weight, collision_scale, ctrl_weight = weights

    num_existing_receding_horizon_samples = len(os.listdir(out_dir)) // 2

    for sample_id in tqdm(range(num_existing_receding_horizon_samples, num_existing_samples), total=num_existing_samples - num_existing_receding_horizon_samples, desc="Generating receding horizon trajectories"):
        json_file = json_files[sample_id]
        with open(json_file, 'r') as f:
            sample_data = json.load(f)
        init_positions = sample_data["init_positions"]
        target_positions = sample_data["target_positions"]
        n_agents = len(init_positions)
        agents = [PointAgent(dt, x_dim=x_dim, u_dim=u_dim, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=ctrl_weight, device=device) for _ in range(n_agents)]
        for agent in agents:
            agent.create_loss_function_mask()
        initial_states = [jnp.array([init_positions[i][0], init_positions[i][1], 0.0, 0.0]) for i in range(n_agents)]
        reference_trajectories = jnp.array([jnp.linspace(jnp.array(init_positions[i]), jnp.array(target_positions[i]), tsteps) for i in range(n_agents)])

        state_trajs, control_trajs, _ = solve_by_horizon(
            agents,
            initial_states,
            reference_trajectories,
            num_iters,
            planning_horizon,
            u_dim,
            tsteps,
            mask_horizon,
            mask_threshold,
            step_size,
            None,   # model
            None,   # model_state
            model_type,
            device,
            collision_weight=collision_weight,
            collision_scale=collision_scale,
        )

        # Convert to proper format for saving
        init_pos_array = jnp.array([[initial_states[i][0], initial_states[i][1]] for i in range(n_agents)])
        target_pos_array = jnp.array(target_positions)
        
        sample_data = save_trajectory_sample(
            sample_id,
            n_agents,
            tsteps,
            dt,
            init_pos_array,
            target_pos_array,
            state_trajs,
            control_trajs
        )

        json_filename = f"receding_horizon_sample_{sample_id:03d}.json"
        json_path = os.path.join(out_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2)

        plot_filename = f"receding_horizon_sample_{sample_id:03d}.png"
        plot_path = os.path.join(out_dir, plot_filename)
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
    planning_horizon = config.game.T_receding_horizon_planning
    mask_horizon = config.game.T_observation
    mask_threshold = config.testing.receding_horizon.mask_threshold

    # Optimization parameters - get agent-specific config
    agent_type = config.game.agent_type
    opt_config = getattr(config.optimization, agent_type)
    num_iters = opt_config.num_iters
    step_size = opt_config.step_size
    Q = jnp.diag(jnp.array(opt_config.Q))
    R = jnp.diag(jnp.array(opt_config.R))
    weights = (opt_config.collision_weight, opt_config.collision_scale, opt_config.control_weight)

    print(f"Configuration loaded:")
    print(f"  N agents: {n_agents}")
    print(f"  Time steps: {tsteps}, dt: {dt}")
    print(f"  Optimization: {num_iters} iters, step size: {step_size}")
    print(f"  Boundary size: {boundary_size}")

    x_dim = opt_config.state_dim
    u_dim = opt_config.control_dim

    # create output directory
    gen_type = config.reference_generation.reference_gen_type
    if gen_type == "fixed":
        output_dir = os.path.join("src/data", f"receding_horizon_trajectories_{n_agents}p")
    elif gen_type == "variable":
        output_dir = os.path.join("src/data", f"receding_horizon_trajectories_upto_{n_agents}p")
    else:
        raise ValueError(f"Invalid generation type: {gen_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    num_samples = config.reference_generation.num_samples

    # generate reference trajectories
    args = {
        "config": config,
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
        "gen_type": gen_type,
        "weights": weights,
        "x_dim": x_dim,
        "u_dim": u_dim,
        "dt": dt,
        "planning_horizon": planning_horizon,
        "mask_horizon": mask_horizon,
        "mask_threshold": mask_threshold,
    }
    generate_receding_horizon_trajectories(**args)