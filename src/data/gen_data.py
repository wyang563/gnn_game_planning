from sim_solver import Simulator
from utils.utils import load_config, parse_arguments
import jax.numpy as jnp
import random

if __name__ == "__main__":
    DEBUG = False

    args = parse_arguments()
    
    # Load configuration from YAML file
    try:
        config = load_config(args.config)
        print(f"=== Testing Receding Horizon Nash Game Simulator ===")
        print(f"Using config file: {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Extract simulation parameters from config
    simulator_config = config['simulator']
    N = simulator_config['n_agents']  # Number of agents
    horizon = simulator_config['horizon']  # Optimization horizon (T steps)
    dt = simulator_config['dt']  # Time step
    init_arena_range = tuple(simulator_config['init_arena_range'])  # Initial position range
    goal_threshold = simulator_config['goal_threshold']  # Convergence threshold
    device = simulator_config['device']
    optimization_iters = simulator_config['optimization_iters']  # Total simulation steps
    step_size = simulator_config['step_size']
    init_type = simulator_config['init_type']
    
    # extracting masking method config
    masking_config = config['masking']
    limit_past_horizon = masking_config['limit_past_horizon']
    masking_method= masking_config['masking_method']
    top_k = masking_config['top_k']
    critical_radius = masking_config['critical_radius']
    mask_horizon = masking_config['mask_horizon']

    # Cost weights (position, velocity, control)
    Q = jnp.diag(jnp.array(simulator_config['Q']))  # Higher position weights
    R = jnp.diag(jnp.array(simulator_config['R']))
    W = jnp.array(simulator_config['W'])  # w1 = collision, w2 = collision cost exp decay, w3 = control, w4 = navigation
    time_steps = int(simulator_config['time_steps'])


    # CHANGE THESE: gen_data_configs
    num_runs = 3
    model_type = "mlp"
    mode = "test"
    gen_data_configs = None

    if model_type == "mlp":
        gen_data_configs = {
            "model_type": model_type,
            "inputs_file": f"src/data/mlp_n_agents_{N}_{mode}/inputs.zarr",
            "x0s_file": f"src/data/mlp_n_agents_{N}_{mode}/x0s.zarr",
            "ref_trajs_file": f"src/data/mlp_n_agents_{N}_{mode}/ref_trajs.zarr",
            "targets_file": f"src/data/mlp_n_agents_{N}_{mode}/targets.zarr",
        }
    elif model_type == "gnn":
        raise NotImplementedError("GNN data generation not implemented yet")

    # Create simulator    
    for _ in range(num_runs):
        # change arena size to be random sized
        mag_size = random.uniform(3.0, 5.0)
        init_arena_range = (-mag_size, mag_size)

        simulator = Simulator(
            n_agents=N,
            Q=Q,
            R=R,
            W=W,
            horizon=horizon,
            mask_horizon=mask_horizon,
            time_steps=time_steps,
            dt=dt,
            init_arena_range=init_arena_range,
            device=device,
            step_size=step_size,
            goal_threshold=goal_threshold,
            optimization_iters=optimization_iters,
            init_type=init_type,
            limit_past_horizon=limit_past_horizon,
            masking_method=masking_method,
            top_k=top_k,
            critical_radius=critical_radius,
            debug=DEBUG,
        )
        print(f"Running simulation {_ + 1} of {num_runs}")
        simulator.run_test(gen_data_configs=gen_data_configs)
