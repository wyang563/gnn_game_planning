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

def solve_by_horizon(
    agents: list[PointAgent],
    initial_states: jnp.ndarray,
    ref_trajs: jnp.ndarray,
    num_iters: int,
    u_dim: int,
    tsteps: int,
    mask_horizon: int,
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

    for iter in tqdm(range(tsteps + 1)):
        # setup horizon arrays
        x_trajs, _, _ = jit_batched_linearize_dyn(init_states, control_trajs)
        horizon_x0s = x_trajs[:, iter]
        horizon_u_trajs = jnp.zeros((n_agents, tsteps, u_dim))

        start_ind = iter
        end_ind = min(start_ind + tsteps, tsteps)
        horizon_ref_trajs = ref_trajs[:, start_ind:end_ind, :]
        if horizon_ref_trajs.shape[1] < tsteps:
            padding = jnp.tile(horizon_ref_trajs[:, -1:], (1, tsteps - horizon_ref_trajs.shape[1], 1))
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

        # batch past x_trajs
        batch_past_x_trajs = [past_x_trajs]
        for i in range(1, n_agents):
            # place index i at start of past_x_trajs
            temp = past_x_trajs[:, 0]
            past_x_trajs = past_x_trajs.at[:, 0].set(past_x_trajs[:, i])
            past_x_trajs = past_x_trajs.at[:, i].set(temp)
            batch_past_x_trajs.append(past_x_trajs)

        batch_past_x_trajs = jnp.array(batch_past_x_trajs)
        masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)

        # game solving optimization loop 
        for _ in range(num_iters + 1):
            horizon_x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(horizon_x0s, horizon_u_trajs)
            all_x_pos = jnp.broadcast_to(horizon_x_trajs[None, :, :, :2], (n_agents, n_agents, tsteps, 2))
            other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))
            mask_for_step = jnp.tile(masks[:, None, :], (1, tsteps, 1))
            a_trajs, b_trajs = jit_batched_linearize_loss(horizon_x_trajs, horizon_u_trajs, horizon_ref_trajs, other_x_trajs, mask_for_step)
            v_trajs, _ = jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
            horizon_u_trajs += step_size * v_trajs

        # update u_trajs
        control_trajs = control_trajs.at[:, iter, :].set(horizon_u_trajs[:, 0, :])
    
    # calculate final x_trajs
    final_x_trajs, _, _ = jit_batched_linearize_dyn(init_states, control_trajs)
    return final_x_trajs, control_trajs

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

    # Optimization parameters
    num_iters = config.optimization.num_iters
    step_size = config.optimization.step_size
    collision_weight = config.optimization.collision_weight
    collision_scale = config.optimization.collision_scale
    control_weight = config.optimization.control_weight
    Q = jnp.diag(jnp.array(config.optimization.Q))
    R = jnp.diag(jnp.array(config.optimization.R))

    # genera random inits
    boundary_size = 3
    init_ps, goals = random_init(n_agents, (-boundary_size, boundary_size))
    init_ps = jnp.array([jnp.array([init_ps[i][0], init_ps[i][1], 0.0, 0.0]) for i in range(n_agents)])
    agents = [PointAgent(dt, x_dim=4, u_dim=2, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=control_weight, device=device) for _ in range(n_agents)]

    # setup loss functions
    for agent in agents:
        agent.create_loss_function_mask()

    ref_trajs = jnp.array([jnp.linspace(init_ps[i][:2], goals[i], tsteps) for i in range(n_agents)])
    mask_horizon = config.game.T_receding_horizon_planning
    u_dim = 2
    model_type = "gnn"
    model_path = "log/gnn_full_planning_true_goals_maxN_10_T_50_obs_10_lr_0.001_bs_32_sigma1_0.03_sigma2_0.03_epochs_100/20251024_152912/psn_best_model.pkl"
    model, model_state = load_trained_gnn_models(model_path, config.gnn.obs_input_type)

    # solve by horizon
    final_x_trajs, control_trajs = solve_by_horizon(agents, init_ps, ref_trajs, num_iters, u_dim, tsteps, mask_horizon, step_size, model, model_state, model_type, device)







