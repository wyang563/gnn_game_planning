import pandas as pd
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.plot import plot_point_agent_trajs, plot_point_agent_gif
from load_config import load_config, setup_jax_config, get_device_config
from solver.point_agent import PointAgent
from models.train_gnn import load_trained_gnn_models
from solver.solve_by_horizon import solve_by_horizon

def load_pedestrian_data(csv_path: str):
    """
    Load pedestrian trajectory data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing pedestrian trajectories
        
    Returns:
        jnp.ndarray: Array of shape [n_agents, time_steps, 4] where the last
                     dimension contains [px, py, vx, vy]
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique agent IDs and sort them
    agent_ids = sorted(df['id'].unique())
    n_agents = len(agent_ids)
    
    # Get the number of time steps (assuming all agents have the same number)
    time_steps = df[df['id'] == agent_ids[0]].shape[0]
    
    # Initialize the output array
    data = np.zeros((n_agents, time_steps, 4))
    
    # Fill in the data for each agent
    for i, agent_id in enumerate(agent_ids):
        # Get data for this agent and sort by frame
        agent_data = df[df['id'] == agent_id].sort_values('frame')
        
        # Extract position and velocity data
        data[i, :, 0] = agent_data['x_est'].values  # px
        data[i, :, 1] = agent_data['y_est'].values  # py
        data[i, :, 2] = agent_data['vx_est'].values  # vx
        data[i, :, 3] = agent_data['vy_est'].values  # vy
    
    # Convert to JAX numpy array
    return jnp.array(data)

if __name__ == "__main__":
    # load config
    config = load_config()
    setup_jax_config()
    device = get_device_config()
    print(f"Using device: {device}")

    # Extract parameters from configuration
    dt = config.game.dt
    tsteps = config.game.T_total
    n_agents = config.game.N_agents  
    init_type = config.game.initiation_type
    mask_threshold = config.testing.receding_horizon.mask_threshold
    planning_horizon = config.game.T_receding_horizon_planning

    # Optimization parameters
    num_iters = config.optimization.num_iters
    step_size = config.optimization.step_size
    collision_weight = config.optimization.collision_weight
    collision_scale = config.optimization.collision_scale
    control_weight = config.optimization.control_weight
    Q = jnp.diag(jnp.array(config.optimization.Q))
    R = jnp.diag(jnp.array(config.optimization.R))

    csv_path = "src/data/vci-dataset-citr/data/trajectories_filtered/p2p_bi/bidirection_no_vehicle_3v7_01_traj_ped_filtered.csv"
    data = load_pedestrian_data(csv_path)
    n_agents = data.shape[0]

    # get initial positions and setup agents
    init_states = data[:, 0, :]
    init_ps = init_states[:, :2]
    goals = data[:, -1, :2]

    # reconfigured timesteps and other settings
    tsteps = data.shape[1]
    num_iters = 100
    collision_weight = 5.0

    agents = [PointAgent(dt, x_dim=4, u_dim=2, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=control_weight, device=device) for _ in range(n_agents)]
    for agent in agents:
        agent.create_loss_function_mask()
    
    ref_trajs = jnp.array([jnp.linspace(init_ps[i][:2], goals[i], tsteps) for i in range(n_agents)])
    mask_horizon = config.game.T_observation
    u_dim = 2
    mask_mag = None # default definition of mask magnitude
    
    model_type = "gnn"
    model_path = "log/gnn_full_MP_3_edge-metric_full_top-k_5/train_n_agents_10_T_50_obs_10_lr_0.001_bs_32_sigma1_0.75_sigma2_0.75_epochs_50_loss_type_ego_agent_cost/20251105_222834/psn_best_model.pkl"
    model, model_state = load_trained_gnn_models(model_path, config.gnn.obs_input_type)

    final_x_trajs, control_trajs, simulation_masks = solve_by_horizon(
        agents=agents,
        initial_states=init_states,
        ref_trajs=ref_trajs,
        num_iters=num_iters,
        u_dim=u_dim,
        tsteps=tsteps,
        planning_horizon=planning_horizon,
        mask_horizon=mask_horizon,
        mask_threshold=mask_threshold,
        step_size=step_size,
        model=model,
        model_state=model_state,
        model_type=model_type,
        device=device,
        mask_mag=mask_mag,
    )

    plot_point_agent_trajs(final_x_trajs, goals, init_ps, save_path="src/eval/test.png")
    plot_point_agent_gif(final_x_trajs, goals, init_ps, simulation_masks, 0, "src/eval/test.gif")