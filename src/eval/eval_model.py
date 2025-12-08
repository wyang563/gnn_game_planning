import jax
import jax.numpy as jnp
import sys
import os
# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from load_config import load_config, setup_jax_config, get_device_config
from models.train_mlp import load_trained_psn_models
from models.train_gnn import load_trained_gnn_models
from solver.solve_by_horizon import solve_by_horizon
from solver.point_agent import PointAgent
from solver.drone_agent import DroneAgent
from typing import Any, Dict, List
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from eval.compute_metrics import compute_metrics

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load all JSON files from the dataset directory."""
    dataset_dir = Path(dataset_path)
    json_files = sorted(dataset_dir.glob("eval_data_sample_*.json"))
    
    dataset = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            dataset.append(data)
    return dataset

def eval_model(
    **kwargs: Any
) -> None:

    model_path = kwargs["model_path"]
    model_type = kwargs["model_type"]
    dataset_path = kwargs["dataset_path"]
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
    top_k_mask = kwargs.get("top_k_mask", 3)
    pos_dim = kwargs.get("pos_dim", 2)
    eval_all_methods = kwargs.get("eval_all_methods", False)
    test_mode = kwargs.get("test_mode", False)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Define all methods to evaluate
    if eval_all_methods:
        methods = ["nearest_neighbors", "jacobian", "cost_evolution", "barrier_function"]
    else:
        methods = []
    
    # Load model if provided
    if model_path is not None:
        if model_type == "mlp":
            model, model_state = load_trained_psn_models(model_path, model_type)
            methods.append("model_mlp")
        elif model_type == "gnn":
            model, model_state = load_trained_gnn_models(model_path, model_type)
            methods.append("model_gnn")
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        model = None
        model_state = None
    
    # Store results for all methods
    all_results = {method: [] for method in methods}
    
    # Evaluate each sample
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
        n_agents = sample["n_agents"]
        init_ps_raw = jnp.array(sample["init_ps"])
        goals = jnp.array(sample["init_goals"])

        # modify tsteps based on n_agents to allow for more flexible route planning when there are more agents and the arena is bigger
        tsteps = max(50, n_agents * 4) 
        
        # add velocity to initial positions
        init_ps = jnp.array([jnp.array(init_ps_raw[i][:pos_dim].tolist() + [0.0] * pos_dim) for i in range(n_agents)])
        
        # Create agents
        agent_class = PointAgent if agent_type == "point" else DroneAgent
        agents = [
            agent_class(
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
            agent.create_loss_function_mask()
        
        # Create reference trajectories
        ref_trajs = jnp.array([jnp.linspace(init_ps[i][:pos_dim], goals[i], tsteps) for i in range(n_agents)])
        
        # Evaluate each method

        # first run all players in horizon
        method_type = "all"
        method_model = None
        method_state = None

        all_players_x_trajs, _, _ = solve_by_horizon(
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
            model=method_model,
            model_state=method_state,
            model_type=method_type,
            device=device,
            dt=dt,
            use_only_ego_masks=False,
            collision_weight=collision_weight,
            collision_scale=collision_scale,
            disable_tqdm=True,
            top_k_mask=top_k_mask,
            pos_dim=pos_dim,
        )

        for method in methods:
            try:
                # Determine model type for solve_by_horizon
                if method == "model_mlp":
                    method_type = "mlp"
                    method_model = model
                    method_state = model_state
                elif method == "model_gnn":
                    method_type = "gnn"
                    method_model = model
                    method_state = model_state
                else:
                    method_type = method
                    method_model = None
                    method_state = None
                
                # Solve by horizon
                final_x_trajs, control_trajs_result, simulation_masks = solve_by_horizon(
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
                    model=method_model,
                    model_state=method_state,
                    model_type=method_type,
                    device=device,
                    dt=dt,
                    use_only_ego_masks=False,
                    collision_weight=collision_weight,
                    collision_scale=collision_scale,
                    disable_tqdm=True,
                    top_k_mask=top_k_mask,
                    pos_dim=pos_dim,
                )
                
                # Compute metrics
                metrics = compute_metrics(
                    final_x_trajs, 
                    control_trajs_result, 
                    simulation_masks, 
                    all_players_x_trajs,
                    ref_trajs,
                    observation_horizon=mask_horizon
                )
                
                # Add sample metadata
                metrics['sample_idx'] = sample_idx
                metrics['n_agents'] = n_agents
                metrics['method'] = method
                
                all_results[method].append(metrics)
                
            except Exception as e:
                print(f"\nError on sample {sample_idx} with method {method}: {e}")
                import traceback
                traceback.print_exc()
        
        if test_mode and sample_idx >= 5:
            break
    
    # Aggregate results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for method in methods:
        if len(all_results[method]) == 0:
            continue
            
        print(f"\n{method.upper()}:")
        print("-" * 80)
        
        # Convert to DataFrame for easy aggregation
        df = pd.DataFrame(all_results[method])
        
        # Compute mean and std for each metric
        metric_names = ['ADE', 'FDE', 'Nav_Cost', 'Col_Cost', 'Ctrl_Cost', 
                       'Traj_Heading', 'Traj_Length', 'Min_Dist', 'Consistency', 
                       'Avg_Num_Players_Selected']
        
        for metric in metric_names:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"  {metric:30s}: {mean_val:10.4f} ± {std_val:10.4f}")
    
    # Save results to CSV
    model_dir = Path(model_path).parent
    output_dir = model_dir / "eval_results"
    output_dir.mkdir(exist_ok=True)
    
    for method in methods:
        if len(all_results[method]) > 0:
            df = pd.DataFrame(all_results[method])
            output_file = output_dir / f"results_{method}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved {method} results to {output_file}")
    
    # Save summary statistics
    summary_data = []
    for method in methods:
        if len(all_results[method]) > 0:
            df = pd.DataFrame(all_results[method])
            summary = {'method': method}
            for metric in metric_names:
                if metric in df.columns:
                    summary[f'{metric}_mean'] = df[metric].mean()
                    summary[f'{metric}_std'] = df[metric].std()
            summary_data.append(summary)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary statistics to {summary_file}")
        
        # Also save as pretty printed table
        summary_txt = output_dir / "summary_statistics.txt"
        with open(summary_txt, 'w') as f:
            f.write(summary_df.to_string(index=False))
        print(f"Saved summary table to {summary_txt}")

if __name__ == "__main__":
    config = load_config()
    setup_jax_config()
    device = get_device_config()
    print(f"Using device: {device}")

    # TOGGLE FOR EVALUATING ALL METHODS
    eval_all_methods = True 
    
    # Extract parameters from configuration
    dt = config.game.dt
    tsteps = config.game.T_total
    n_agents = config.game.N_agents  
    init_type = config.game.initiation_type
    mask_threshold = config.testing.receding_horizon.mask_threshold
    planning_horizon = config.game.T_receding_horizon_planning
    mask_horizon = config.game.T_observation

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

    # Model configuration - set to None to only evaluate baselines, or provide path for model evaluation
    model_path = "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.7_sigma2_0.7_sigma3_1.5_noise_std_0.5_epochs_30_loss_type_ego_agent_cost/20251205_094622/psn_best_model.pkl"
    model_type = "gnn"  
    dataset_path = f"src/data/{agent_type}_agent_data/eval_data_upto_20p"
    top_k_mask = 1

    TEST_MODE = False 

    # multiple model paths
    # model_paths = [
    #     "log/point_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.11_sigma2_0.11_sigma3_0.01_noise_std_0.5_epochs_30_loss_type_similarity/20251207_125719/psn_best_model.pkl",
    #     "log/point_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_1.0_sigma2_1.0_sigma3_0.1_noise_std_0.5_epochs_30_loss_type_ego_agent_cost/20251207_125802/psn_best_model.pkl",
    #     "log/point_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.0003_bs_32_sigma1_1.0_sigma2_1.0_sigma3_0.5_noise_std_0.5_epochs_30_loss_type_ego_agent_cost/20251206_232715/psn_best_model.pkl",
    #     "log/point_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.0003_bs_32_sigma1_0.08_sigma2_0.08_sigma3_0.04_noise_std_0.5_epochs_30_loss_type_similarity/20251206_173412/psn_best_model.pkl",
    #     "log/point_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.0003_bs_32_sigma1_0.05_sigma2_0.05_sigma3_0.02_noise_std_0.5_epochs_30_loss_type_similarity/20251206_232630/psn_best_model.pkl"
    # ]

    model_paths = [
        "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.7_sigma2_0.7_sigma3_1.5_noise_std_0.1_epochs_30_loss_type_ego_agent_cost/20251127_091515/psn_best_model.pkl",
        "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.7_sigma2_0.7_sigma3_1.5_noise_std_0.5_epochs_30_loss_type_ego_agent_cost/20251205_094652/psn_best_model.pkl",
        "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.11_sigma2_0.11_sigma3_0.25_noise_std_0.1_epochs_30_loss_type_similarity/20251127_091623/psn_best_model.pkl",
        "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.7_sigma2_0.7_sigma3_1.5_noise_std_0.5_epochs_30_loss_type_ego_agent_cost/20251205_094622/psn_best_model.pkl",
        "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.11_sigma2_0.11_sigma3_0.25_noise_std_0.5_epochs_30_loss_type_similarity/20251203_144120/psn_best_model.pkl"
    ]

    for i, model_path in enumerate(model_paths):
        top_k_mask = i + 1

        print("RUNNING EVALUATIONS")
        print("=" * 60)
        print(f"  agent_type: {agent_type}")
        print(f"  model_path: {model_path}")
        print(f"  model_type: {model_type}")
        print(f"  dataset_path: {dataset_path}")
        print(f"  top_k_mask: {top_k_mask}")
        print(f"  eval_all_methods: {eval_all_methods}")
        print(f"  test_mode: {TEST_MODE}")
        print(f"  num_iters: {num_iters}")
        print(f"  step_size: {step_size}")
        print(f"  collision_weight: {collision_weight}")
        print(f"  collision_scale: {collision_scale}")
        print(f"  control_weight: {control_weight}")
        print(f"  Q: {Q}")
        print(f"  R: {R}")
        print(f"  dt: {dt}")
        print(f"  tsteps: {tsteps}")
        print(f"  planning_horizon: {planning_horizon}")
        print(f"  top_k_mask: {top_k_mask}")

        args = {
            "model_path": model_path,
            "model_type": model_type,
            "dataset_path": dataset_path,
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
            "u_dim": opt_config.control_dim,
            "x_dim": opt_config.state_dim,
            "dt": dt, 
            "top_k_mask": top_k_mask,
            "pos_dim": opt_config.state_dim // 2,
            "eval_all_methods": eval_all_methods,
            "test_mode": TEST_MODE,
        }

        eval_model(**args)
