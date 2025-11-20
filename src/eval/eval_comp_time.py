import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from load_config import load_config, setup_jax_config, get_device_config
from models.train_mlp import load_trained_psn_models
from models.train_gnn import load_trained_gnn_models
from solver.solve_by_horizon import solve_by_horizon_sequential
from tqdm import tqdm
import jax.numpy as jnp
import time
from typing import Any
from eval_model import load_dataset
from solver.point_agent import PointAgent
from utils.goal_init import random_init

def eval_comp_time(**kwargs: Any) -> None:
    model_path = kwargs["model_path"]
    model_type = kwargs["model_type"]
    num_iters = kwargs["num_iters"]
    step_size = kwargs["step_size"]
    collision_weight = kwargs["collision_weight"]
    collision_scale = kwargs["collision_scale"]
    control_weight = kwargs["control_weight"]
    Q = kwargs["Q"]
    R = kwargs["R"]
    tsteps = kwargs["tsteps"]
    planning_horizon = kwargs["planning_horizon"]
    mask_horizon = kwargs["mask_horizon"]
    mask_threshold = kwargs["mask_threshold"]
    device = kwargs["device"]
    u_dim = kwargs.get("u_dim", 2)
    x_dim = kwargs.get("x_dim", 4)
    dt = kwargs.get("dt", 0.1)
    eval_num_agents = kwargs.get("eval_num_agents", [5, 10, 20, 50])

    for num_agents in eval_num_agents:
        print(f"Evaluating {num_agents} agents...")
        opt_times_by_method = {"all": [], "gnn": []}
        for _ in range(2):
            # increment number of steps by number of agents
            boundary_size = num_agents**(0.7)  * 1.75
            if 0 < num_agents <= 10:
                tsteps = 50
            elif 10 < num_agents <= 20:
                tsteps = 100 
            else:
                tsteps = 150
            
            # generate random inits
            init_ps, init_goals = random_init(num_agents, (-boundary_size, boundary_size))
            init_ps = jnp.array([jnp.array([init_ps[i][0], init_ps[i][1], 0.0, 0.0]) for i in range(num_agents)])

            agents = [
                PointAgent(
                    dt, x_dim=x_dim, u_dim=u_dim, Q=Q, R=R, 
                    collision_weight=collision_weight, 
                    collision_scale=collision_scale, 
                    ctrl_weight=control_weight, 
                    device=device
                ) 
                for _ in range(n_agents)
            ]
            
            # Setup loss functions
            for agent in agents:
                agent.create_loss_functions_no_mask()
            
            ref_trajs = jnp.array([jnp.linspace(init_ps[i][:2], init_goals[i], tsteps) for i in range(num_agents)])

            for method_type in ["all", "gnn"]:
                # Load model if provided
                if model_path is not None and method_type == "gnn":
                    if model_type == "mlp":
                        model, model_state = load_trained_psn_models(model_path, model_type)
                    elif model_type == "gnn":
                        model, model_state = load_trained_gnn_models(model_path, model_type)
                    else:
                        raise ValueError(f"Invalid model type: {model_type}")
                else:
                    model = None
                    model_state = None

                _, _, _, avg_optimization_time = solve_by_horizon_sequential(
                    agents=agents,
                    initial_states=init_ps,
                    ref_trajs=ref_trajs,
                    num_iters=num_iters,
                    planning_horizon=planning_horizon,
                    u_dim=u_dim,
                    tsteps=tsteps,
                    mask_horizon=mask_horizon,
                    mask_threshold=mask_threshold,
                    step_size=step_size,
                    model=model,
                    model_state=model_state,
                    model_type=method_type,
                    device=device,
                    dt=dt,
                    use_only_ego_masks=False,
                    collision_weight=collision_weight,
                    collision_scale=collision_scale,
                )

                opt_times_by_method[method_type].append(avg_optimization_time)

        # print results for num agents
        print(f"Results for {num_agents} agents:")
        print(f"All players: {sum(opt_times_by_method['all']) / len(opt_times_by_method['all'])}")
        print(f"GNN: {sum(opt_times_by_method['gnn']) / len(opt_times_by_method['gnn'])}")

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
    mask_horizon = config.game.T_observation

    # Optimization parameters
    num_iters = config.optimization.num_iters
    step_size = config.optimization.step_size
    collision_weight = config.optimization.collision_weight
    collision_scale = config.optimization.collision_scale
    control_weight = config.optimization.control_weight
    Q = jnp.diag(jnp.array(config.optimization.Q))
    R = jnp.diag(jnp.array(config.optimization.R))

    # Model configuration - set to None to only evaluate baselines, or provide path for model evaluation
    model_path = "log/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_10_T_50_obs_10_lr_0.0003_bs_32_sigma1_0.11_sigma2_0.11_epochs_50_loss_type_similarity/20251110_201039/psn_best_model.pkl"
    model_type = "gnn"  

    eval_num_agents = [5, 10, 20, 50]

    args = {
        "model_path": model_path,
        "model_type": model_type,
        "num_iters": num_iters,
        "step_size": step_size,
        "collision_weight": collision_weight,
        "collision_scale": collision_scale,
        "control_weight": control_weight,
        "Q": Q,
        "R": R,
        "dt": dt,
        "tsteps": tsteps,
        "planning_horizon": planning_horizon,
        "mask_horizon": mask_horizon,
        "mask_threshold": mask_threshold,
        "device": device,
        "u_dim": 2,
        "x_dim": 4,
        "dt": dt, 
        "eval_num_agents": eval_num_agents,
    }

    eval_comp_time(**args)
