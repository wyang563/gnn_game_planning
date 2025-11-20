import os
from load_config import load_config, setup_jax_config, get_device_config
from models.train_mlp import load_trained_psn_models
from models.train_gnn import load_trained_gnn_models
from solver.solve_by_horizon import solve_by_horizon
from tqdm import tqdm
import jax.numpy as jnp
import time

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
    dataset_path = "src/data/eval_data_upto_20p"
