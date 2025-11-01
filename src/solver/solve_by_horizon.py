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
from solver.solve import create_batched_loss_functions_mask, solve_ilqgames_parallel_mask
from flax import linen as nn
from utils.goal_init import random_init
from models.train_gnn import GNNSelectionNetwork, load_trained_gnn_models
from models.train_mlp import PlayerSelectionNetwork, load_trained_psn_models
from tqdm import tqdm
from utils.plot import plot_trajs, plot_agent_gif

def solve_by_horizon(
    agents: list[PointAgent],
    initial_states: jnp.ndarray,
    ref_trajs: jnp.ndarray,
    num_iters: int,
    planning_horizon: int,
    u_dim: int,
    tsteps: int,
    mask_horizon: int,
    mask_threshold: float,
    step_size: float,
    model: nn.Module,
    model_state: Any,
    model_type: str,
    device: Any
) -> None:
    n_agents = len(agents)

    # create batched functions
    jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss = create_batched_loss_functions_mask(agents, device)

    # initialize batched arrays
    control_trajs = jnp.zeros((n_agents, tsteps, u_dim))
    init_states = jnp.array([initial_states[i] for i in range(n_agents)])

    # logging data
    simulation_masks = []

    for iter in tqdm(range(tsteps + 1)):
        # setup horizon arrays
        x_trajs, _, _ = jit_batched_linearize_dyn(init_states, control_trajs)
        horizon_x0s = x_trajs[:, iter]
        horizon_u_trajs = jnp.zeros((n_agents, planning_horizon, u_dim))

        start_ind = min(iter, tsteps - 1)
        end_ind = min(start_ind + planning_horizon, tsteps)
        horizon_ref_trajs = ref_trajs[:, start_ind:end_ind, :]
        if horizon_ref_trajs.shape[1] < planning_horizon:
            padding = jnp.tile(horizon_ref_trajs[:, -1:], (1, planning_horizon - horizon_ref_trajs.shape[1], 1))
            horizon_ref_trajs = jnp.concatenate([horizon_ref_trajs, padding], axis=1)

        # get mask horizon input trajs
        start_ind = max(0, iter - mask_horizon)
        end_ind = max(iter, 1)

        # get past x_trajs
        past_x_trajs = x_trajs[:, start_ind:end_ind, :]
        if past_x_trajs.shape[1] < mask_horizon:
            padding = jnp.tile(past_x_trajs[:, -1:, :], (1, mask_horizon - past_x_trajs.shape[1], 1))
            past_x_trajs = jnp.concatenate([past_x_trajs, padding], axis=1)
        
        # get mask based off past x_trajs
        if model_type == "mlp":
            past_x_trajs = past_x_trajs.reshape(n_agents, -1)
        elif model_type == "gnn":
            past_x_trajs = past_x_trajs.transpose(1, 0, 2)
        elif model_type == "all":
            pass

        batch_past_x_trajs = past_x_trajs[None, ...]
        
        # calculate ego masks based off model player selection type
        if model_type in ["mlp", "gnn"]:
            ego_masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
            ego_masks = ego_masks[0]
            if model_type == "mlp":
                final_masks_mlp = jnp.zeros((n_agents))
                ego_masks = final_masks_mlp.at[1:].set(ego_masks)
        elif model_type == "all":
            ego_masks = jnp.ones((n_agents))
            ego_masks = ego_masks.at[0].set(0.0)

        # threshold ego masks
        ego_masks = jnp.where(ego_masks > mask_threshold, 1.0, 0.0)
        simulation_masks.append(ego_masks)

        masks = ~(jnp.eye(n_agents).astype(jnp.bool_))
        masks = masks.astype(jnp.float32)
        # Set the first row of the masks matrix to be ego_masks
        masks = masks.at[0].set(ego_masks)

        # game solving optimization loop 
        for _ in range(num_iters + 1):
            horizon_x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(horizon_x0s, horizon_u_trajs)
            all_x_pos = jnp.broadcast_to(horizon_x_trajs[None, :, :, :2], (n_agents, n_agents, planning_horizon, 2))
            other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))
            mask_for_step = jnp.tile(masks[:, None, :], (1, planning_horizon, 1))
            a_trajs, b_trajs = jit_batched_linearize_loss(horizon_x_trajs, horizon_u_trajs, horizon_ref_trajs, other_x_trajs, mask_for_step)
            v_trajs, _ = jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
            horizon_u_trajs += step_size * v_trajs

        # update u_trajs
        control_trajs = control_trajs.at[:, iter, :].set(horizon_u_trajs[:, 0, :])
    
    # calculate final x_trajs
    final_x_trajs, _, _ = jit_batched_linearize_dyn(init_states, control_trajs)
    return final_x_trajs, control_trajs, simulation_masks

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
    mask_threshold = config.testing.receding_horizon.mask_threshold
    planning_horizon = config.game.T_receding_horizon_planning

    # redefinitions
    n_agents = 25
    tsteps = 100

    # Optimization parameters
    num_iters = config.optimization.num_iters
    step_size = config.optimization.step_size
    collision_weight = config.optimization.collision_weight
    collision_scale = config.optimization.collision_scale
    control_weight = config.optimization.control_weight
    Q = jnp.diag(jnp.array(config.optimization.Q))
    R = jnp.diag(jnp.array(config.optimization.R))

    # genera random inits
    boundary_size = 3.5
    init_ps, goals = random_init(n_agents, (-boundary_size, boundary_size))
    init_ps = jnp.array([jnp.array([init_ps[i][0], init_ps[i][1], 0.0, 0.0]) for i in range(n_agents)])
    agents = [PointAgent(dt, x_dim=4, u_dim=2, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=control_weight, device=device) for _ in range(n_agents)]

    # setup loss functions
    for agent in agents:
        agent.create_loss_function_mask()

    ref_trajs = jnp.array([jnp.linspace(init_ps[i][:2], goals[i], tsteps) for i in range(n_agents)])
    mask_horizon = config.game.T_observation
    u_dim = 2

    # model_type = "mlp"
    # model_path = "log/psn_gru_full_planning_true_goals_N_10_T_50_obs_10_lr_0.002_bs_64_sigma1_0.075_sigma2_0.075_epochs_50/20251023_001904/psn_best_model.pkl"
    # model, model_state = load_trained_psn_models(model_path, config.psn.obs_input_type)

    model_type = "gnn"
    model_path = "log/gnn_full_planning_true_goals_maxN_10_T_50_obs_10_lr_0.001_bs_32_sigma1_0.03_sigma2_0.03_epochs_50_mp_3_loss_type_similarity/20251030_125343/psn_best_model.pkl"
    model, model_state = load_trained_gnn_models(model_path, config.gnn.obs_input_type)

    # solve by horizon
    final_x_trajs, control_trajs, simulation_masks = solve_by_horizon(
        agents=agents,
        initial_states=init_ps,
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
        device=device
    )
    plot_trajs(final_x_trajs, goals, init_ps, save_path="src/solver/test.png")
    plot_agent_gif(final_x_trajs, goals, init_ps, simulation_masks, 0, "src/solver/test.gif")







