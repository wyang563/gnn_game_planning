import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt
from jax import vmap, jit, grad
from datetime import datetime

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
from utils.goal_init import random_init, origin_init_collision
from models.train_gnn import GNNSelectionNetwork, load_trained_gnn_models
from models.train_mlp import PlayerSelectionNetwork, load_trained_psn_models
from tqdm import tqdm
from utils.plot import plot_point_agent_trajs, plot_point_agent_gif
# from eval.baselines import nearest_neighbors, jacobian, cost_evolution, barrier_function
from models.policies import nearest_neighbors_top_k, jacobian_top_k, barrier_function_top_k, cost_evolution_top_k
from utils.agent_selection_utils import agent_type_to_agent_class, agent_type_to_plot_functions
from eval.compute_metrics import compute_minimum_distance 

# this is the sequential version so we can calculate compute performance metrics
def solve_by_horizon_sequential(
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
    device: Any,
    top_k_mask: float = 3,
    dt: float = 0.1,
    use_only_ego_masks: bool = True,
    collision_weight: float = 2.0,
    collision_scale: float = 1.0,
    disable_tqdm: bool = False,
):
    total_game_theory_optimization_time = 0.0
    n_agents = len(agents)
    control_trajs = jnp.zeros((n_agents, tsteps, u_dim))
    init_states = jnp.array([initial_states[i] for i in range(n_agents)])
    
    simulation_masks = []

    for iter in tqdm(range(tsteps + 1), disable=disable_tqdm):
        x_trajs = []
        for i in range(n_agents):
            agent_x_traj, _, _ = agents[i].linearize_dyn(init_states[i], control_trajs[i])
            x_trajs.append(agent_x_traj)
        x_trajs = jnp.stack(x_trajs, axis=0)
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
        match model_type:
            case "mlp" | "gnn":
                masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
                masks = jnp.squeeze(masks, axis=0) # squeeze batch dimension
                if model_type == "mlp":
                    # Transform (n_agents, n_agents-1) mask to (n_agents, n_agents) with 0 at row i, col i since MLP only returns n_agents - 1 sized masks
                    padded_masks = []
                    for i in range(n_agents):
                        mask_row = masks[i]
                        row_with_zero = jnp.concatenate([mask_row[:i], jnp.array([0]), mask_row[i:]])
                        padded_masks.append(row_with_zero)
                    masks = jnp.stack(padded_masks, axis=0)
            case "nearest_neighbors":
                masks = nearest_neighbors_top_k(past_x_trajs, top_k_mask)
            case "jacobian":
                masks = jacobian_top_k(past_x_trajs, top_k_mask, dt=dt, w1=collision_weight, w2=collision_scale)
            case "cost_evolution":
                masks = cost_evolution_top_k(past_x_trajs, top_k=top_k_mask, w1=collision_weight, w2=collision_scale)
            case "barrier_function":
                masks = barrier_function_top_k(past_x_trajs, top_k_mask, R=0.5, kappa=5.0)
            case "all":
                masks = ~(jnp.eye(n_agents).astype(jnp.bool_))
                masks = masks.astype(jnp.float32)
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        # threshold ego masks
        masks = jnp.where(masks > mask_threshold, 1.0, 0.0)

        # assumes ego agent_id is 0
        simulation_masks.append(masks[0])

        # condition for if we want all other agents to consider every other agent
        if use_only_ego_masks:
            ego_masks = masks[0]
            masks = ~(jnp.eye(n_agents).astype(jnp.bool_))
            masks = masks.astype(jnp.float32)
            # Set the first row of the masks matrix to be ego_masks
            masks = masks.at[0].set(ego_masks)
        
        optimization_start_time = time.time()
        for _ in range(num_iters + 1):
            horizon_x_trajs = []
            A_trajs = []
            B_trajs = []
            for agent_id in range(n_agents):
                agent_horizon_x_traj, agent_A_traj, agent_B_traj = agents[agent_id].linearize_dyn(horizon_x0s[agent_id], horizon_u_trajs[agent_id])
                horizon_x_trajs.append(agent_horizon_x_traj)
                A_trajs.append(agent_A_traj)
                B_trajs.append(agent_B_traj)

            horizon_x_trajs = jnp.stack(horizon_x_trajs, axis=0)

            # calcualte other x_trajs based off mask
            a_trajs = []
            b_trajs = []
            for agent_id in range(n_agents):
                agent_other_states = horizon_x_trajs[masks[agent_id] != 0]
                agent_other_states = agent_other_states.transpose(1, 0, 2)
                a_traj, b_traj = agents[agent_id].linearize_loss(horizon_x_trajs[agent_id], horizon_u_trajs[agent_id], horizon_ref_trajs[agent_id], agent_other_states)
                a_trajs.append(a_traj)
                b_trajs.append(b_traj)
            
            control_updates = []
            # calculate optimal control
            for i in range(n_agents):
                v_traj, _ = agents[i].compiled_solve(A_trajs[i], B_trajs[i], a_trajs[i], b_trajs[i])
                control_updates.append(v_traj)

            # update u_trajs
            for i in range(n_agents):
                horizon_u_trajs = horizon_u_trajs.at[i].add(step_size * control_updates[i])


        optimization_end_time = time.time()
        total_game_theory_optimization_time += optimization_end_time - optimization_start_time

        # update u_trajs
        control_trajs = control_trajs.at[:, iter, :].set(horizon_u_trajs[:, 0, :])
    
    # calculate final x_trajs
    final_x_trajs = []
    for agent_id in range(n_agents):
        agent_horizon_x_traj, _, _ = agents[agent_id].linearize_dyn(horizon_x0s[agent_id], horizon_u_trajs[agent_id])
        final_x_trajs.append(agent_horizon_x_traj)
    final_x_trajs = jnp.stack(final_x_trajs, axis=0)

    avg_optimization_time = total_game_theory_optimization_time / (tsteps + 1)
    return final_x_trajs, control_trajs, simulation_masks, avg_optimization_time

# standard parallelized/optimized version of solver
def solve_by_horizon(
    agents: list[PointAgent],
    initial_states: jnp.ndarray,
    ref_trajs: jnp.ndarray,
    num_iters: int,
    planning_horizon: int,
    u_dim: int,
    pos_dim: int,
    tsteps: int,
    mask_horizon: int,
    mask_threshold: float,
    step_size: float,
    model: nn.Module,
    model_state: Any,
    model_type: str,
    device: Any,
    top_k_mask: float = 3,
    dt: float = 0.1,
    use_only_ego_masks: bool = True,
    collision_weight: float = 2.0,
    collision_scale: float = 1.0,
    disable_tqdm: bool = False,
): 
    n_agents = len(agents)

    # create batched functions
    jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, jit_batched_loss = create_batched_loss_functions_mask(agents, device)

    # initialize batched arrays
    control_trajs = jnp.zeros((n_agents, tsteps, u_dim))
    init_states = jnp.array([initial_states[i] for i in range(n_agents)])

    # logging data
    simulation_masks = []

    for iter in tqdm(range(tsteps + 1), disable=disable_tqdm):
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
        match model_type:
            case "mlp" | "gnn":
                masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
                masks = jnp.squeeze(masks, axis=0) # squeeze batch dimension
                if model_type == "mlp":
                    # Transform (n_agents, n_agents-1) mask to (n_agents, n_agents) with 0 at row i, col i since MLP only returns n_agents - 1 sized masks
                    padded_masks = []
                    for i in range(n_agents):
                        mask_row = masks[i]
                        row_with_zero = jnp.concatenate([mask_row[:i], jnp.array([0]), mask_row[i:]])
                        padded_masks.append(row_with_zero)
                    masks = jnp.stack(padded_masks, axis=0)
            case "nearest_neighbors":
                masks = nearest_neighbors_top_k(past_x_trajs, top_k_mask)
            case "jacobian":
                masks = jacobian_top_k(past_x_trajs, top_k_mask, dt=dt, w1=collision_weight, w2=collision_scale)
            case "cost_evolution":
                masks = cost_evolution_top_k(past_x_trajs, top_k=top_k_mask, w1=collision_weight, w2=collision_scale)
            case "barrier_function":
                masks = barrier_function_top_k(past_x_trajs, top_k_mask, R=0.5, kappa=5.0)
            case "all":
                masks = ~(jnp.eye(n_agents).astype(jnp.bool_))
                masks = masks.astype(jnp.float32)
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        # threshold ego masks
        masks = jnp.where(masks > mask_threshold, 1.0, 0.0)

        # assumes ego agent_id is 0
        simulation_masks.append(masks[0])

        # condition for if we want all other agents to consider every other agent
        if use_only_ego_masks:
            ego_masks = masks[0]
            masks = ~(jnp.eye(n_agents).astype(jnp.bool_))
            masks = masks.astype(jnp.float32)
            # Set the first row of the masks matrix to be ego_masks
            masks = masks.at[0].set(ego_masks)

        # game solving optimization loop 
        for _ in range(num_iters + 1):
            horizon_x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(horizon_x0s, horizon_u_trajs)
            all_x_pos = jnp.broadcast_to(horizon_x_trajs[None, :, :, :pos_dim], (n_agents, n_agents, planning_horizon, pos_dim))
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

    # Optimization parameters - get agent-specific config
    agent_type = config.game.agent_type
    opt_config = getattr(config.optimization, agent_type)
    num_iters = opt_config.num_iters
    step_size = opt_config.step_size
    collision_weight = opt_config.collision_weight
    collision_scale = opt_config.collision_scale
    control_weight = opt_config.control_weight
    Q = jnp.diag(jnp.array(opt_config.Q))
    R = jnp.diag(jnp.array(opt_config.R))
    x_dim = opt_config.state_dim
    pos_dim = x_dim // 2
    u_dim = opt_config.control_dim

    # CUSTOM CONFIGS
    tsteps = 100
    n_agents = 10 
    num_iters = 150 

    print("Optimization parameters:")
    print(f"  agent_type: {agent_type}")
    print(f"  tsteps: {tsteps}")
    print(f"  num_iters: {num_iters}")
    print(f"  step_size: {step_size}")
    print(f"  collision_weight: {collision_weight}")
    print(f"  collision_scale: {collision_scale}")
    print(f"  control_weight: {control_weight}")

    # genera random inits
    # boundary_size = 1.75 * n_agents**0.5
    boundary_size = 5.0
    match init_type:
        case "random":
            init_ps, goals = random_init(n_agents, (-boundary_size, boundary_size), dims=pos_dim)
        case "origin":
            init_ps, goals = origin_init_collision(n_agents, (-boundary_size, boundary_size), dims=pos_dim)
        case _:
            raise ValueError(f"Invalid init type: {init_type}")
        
    if x_dim == 4:
        init_ps = jnp.array([jnp.array([init_ps[i][0], init_ps[i][1]] + [0.0] * (pos_dim)) for i in range(n_agents)])
    elif x_dim == 6:
        init_ps = jnp.array([jnp.array([init_ps[i][0], init_ps[i][1], init_ps[i][2]] + [0.0] * (pos_dim)) for i in range(n_agents)])
    else:
        raise ValueError(f"Invalid x_dim: {x_dim}")

    agent_class = agent_type_to_agent_class(agent_type)
    agents = [agent_class(dt, x_dim=x_dim, u_dim=u_dim, Q=Q, R=R, collision_weight=collision_weight, collision_scale=collision_scale, ctrl_weight=control_weight, device=device) for _ in range(n_agents)]

    # setup loss functions
    for agent in agents:
        agent.create_loss_function_mask()

    ref_trajs = jnp.array([jnp.linspace(init_ps[i][:pos_dim], goals[i], tsteps) for i in range(n_agents)])
    mask_horizon = config.game.T_observation
    mask_mag = None # default definition of mask magnitude
    use_only_ego_masks = False

    # model_type = "mlp"
    # model_path = "log/psn_gru_full_planning_true_goals_N_10_T_50_obs_10_lr_0.002_bs_64_sigma1_0.075_sigma2_0.075_epochs_50/20251023_001904/psn_best_model.pkl"
    # model, model_state = load_trained_psn_models(model_path, config.psn.obs_input_type)

    model_type = "gnn"
    model_path = "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.0003_bs_32_sigma1_0.1_sigma2_0.1_epochs_50_loss_type_similarity/20251121_223914/psn_best_model.pkl"
    model, model_state = load_trained_gnn_models(model_path, config.gnn.obs_input_type)
    use_only_ego_masks = False 

    # model_type = "barrier_function"
    # mask_mag = 5
    # model = None
    # model_state = None
    

    # solve by horizon
    final_x_trajs, control_trajs, simulation_masks = solve_by_horizon(
        agents=agents,
        initial_states=init_ps,
        ref_trajs=ref_trajs,
        num_iters=num_iters,
        u_dim=u_dim,
        pos_dim=pos_dim,
        tsteps=tsteps,
        planning_horizon=planning_horizon,
        mask_horizon=mask_horizon,
        mask_threshold=mask_threshold,
        step_size=step_size,
        model=model,
        model_state=model_state,
        model_type=model_type,
        device=device,
        use_only_ego_masks=use_only_ego_masks,
        top_k_mask=mask_mag,
        collision_weight=collision_weight,
        collision_scale=collision_scale,
    )

    # calculate minimum distance
    min_distance = compute_minimum_distance(final_x_trajs, pos_dim=pos_dim)
    print(f"Minimum distance: {min_distance}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"log/solve_by_horizon_run/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    plot_save_path = os.path.join(out_dir, "test.png")
    traj_gif_save_path = os.path.join(out_dir, "traj_test.gif")
    mask_gif_save_path = os.path.join(out_dir, "mask_test.gif")

    plot_functions = agent_type_to_plot_functions(agent_type)

    print(f"Creating Trajectory Plot")
    plot_functions["plot_traj"](final_x_trajs, goals, init_ps, save_path=plot_save_path)
    print(f"Creating Trajectory GIF")
    plot_functions["plot_traj_gif"](final_x_trajs, goals, init_ps, save_path=traj_gif_save_path)
    print(f"Creating Mask GIF")
    plot_functions["plot_mask_gif"](final_x_trajs, goals, init_ps, simulation_masks, ego_agent_id=0, save_path=mask_gif_save_path)

