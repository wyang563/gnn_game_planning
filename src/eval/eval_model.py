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
from typing import Any, Dict, List
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

def compute_metrics(
    x_trajs: jnp.ndarray,
    control_trajs: jnp.ndarray,
    simulation_masks: List[jnp.ndarray],
    ref_trajs: jnp.ndarray,
    observation_horizon: int = 10
) -> Dict[str, float]:
    """
    Compute evaluation metrics from the PSN paper.
    
    Args:
        x_trajs: Trajectory array of shape (N, T, state_dim)
        control_trajs: Control array of shape (N, T, u_dim)
        simulation_masks: List of masks for each timestep
        ref_trajs: Reference trajectories of shape (N, T, 2)
        observation_horizon: Number of observation timesteps
    
    Returns:
        Dictionary of computed metrics
    """
    N, T, state_dim = x_trajs.shape
    K = observation_horizon
    
    metrics = {}
    
    # For ego agent (agent 0)
    ego_traj = x_trajs[0]  # (T, state_dim)
    ego_control = control_trajs[0]  # (T, u_dim)
    ego_ref = ref_trajs[0]  # (T, 2)
    
    # ADE: Average Displacement Error (for prediction phase K:K+T)
    # Since we don't have ground truth, we use the reference trajectory
    pred_positions = ego_traj[K:, :2]
    ref_positions = ego_ref[K:]
    if len(pred_positions) > 0:
        ade = jnp.mean(jnp.linalg.norm(pred_positions - ref_positions, axis=1))
        metrics['ADE'] = float(ade)
    else:
        metrics['ADE'] = 0.0
    
    # FDE: Final Displacement Error
    if len(pred_positions) > 0:
        fde = jnp.linalg.norm(pred_positions[-1] - ref_positions[-1])
        metrics['FDE'] = float(fde)
    else:
        metrics['FDE'] = 0.0
    
    # Navigation Cost: sum of squared distance to reference
    nav_cost = jnp.sum(jnp.linalg.norm(ego_traj[:, :2] - ego_ref, axis=1) ** 2)
    metrics['Nav_Cost'] = float(nav_cost)
    
    # Collision Cost: sum of exponential proximity costs with all other agents
    col_cost = 0.0
    for t in range(T):
        for j in range(1, N):
            dist_sq = jnp.sum((ego_traj[t, :2] - x_trajs[j, t, :2]) ** 2)
            col_cost += jnp.exp(-dist_sq)
    metrics['Col_Cost'] = float(col_cost)
    
    # Control Cost: sum of squared control magnitudes
    ctrl_cost = jnp.sum(jnp.linalg.norm(ego_control, axis=1) ** 2)
    metrics['Ctrl_Cost'] = float(ctrl_cost)
    
    # Trajectory heading change (smoothness): cumulative change in direction
    traj_h = 0.0
    for t in range(2, T):
        if state_dim >= 4:
            # Use velocity if available
            v_curr = ego_traj[t, 2:4]
            v_prev = ego_traj[t-1, 2:4]
            norm_curr = jnp.linalg.norm(v_curr)
            norm_prev = jnp.linalg.norm(v_prev)
            if norm_curr > 1e-6 and norm_prev > 1e-6:
                dir_curr = v_curr / norm_curr
                dir_prev = v_prev / norm_prev
                traj_h += jnp.linalg.norm(dir_curr - dir_prev)
        else:
            # Use position differences
            p_curr = ego_traj[t, :2] - ego_traj[t-1, :2]
            p_prev = ego_traj[t-1, :2] - ego_traj[t-2, :2]
            norm_curr = jnp.linalg.norm(p_curr)
            norm_prev = jnp.linalg.norm(p_prev)
            if norm_curr > 1e-6 and norm_prev > 1e-6:
                dir_curr = p_curr / norm_curr
                dir_prev = p_prev / norm_prev
                traj_h += jnp.linalg.norm(dir_curr - dir_prev)
    metrics['Traj_Heading'] = float(traj_h)
    
    # Trajectory length: total distance traveled
    traj_l = jnp.sum(jnp.linalg.norm(jnp.diff(ego_traj[:, :2], axis=0), axis=1))
    metrics['Traj_Length'] = float(traj_l)
    
    # Minimum distance: safety metric (higher is better)
    min_dist = float('inf')
    for t in range(T):
        for j in range(1, N):
            dist = jnp.linalg.norm(ego_traj[t, :2] - x_trajs[j, t, :2])
            min_dist = min(min_dist, float(dist))
    metrics['Min_Dist'] = min_dist
    
    # Consistency: how stable the mask selection is over time
    if len(simulation_masks) > 1:
        consistency = 0.0
        for t in range(1, len(simulation_masks)):
            mask_diff = jnp.sum(jnp.abs(simulation_masks[t] - simulation_masks[t-1]))
            consistency += 1 - (mask_diff / (N - 1))
        consistency /= (len(simulation_masks) - 1)
        metrics['Consistency'] = float(consistency)
    else:
        metrics['Consistency'] = 1.0
    
    # Additional useful metrics
    metrics['Avg_Num_Players_Selected'] = float(jnp.mean(jnp.array([jnp.sum(m) for m in simulation_masks])))
    
    return metrics


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
    dt = kwargs["dt"]
    tsteps = kwargs["tsteps"]
    planning_horizon = kwargs["planning_horizon"]
    mask_horizon = kwargs["mask_horizon"]
    mask_threshold = kwargs["mask_threshold"]
    device = kwargs["device"]
    u_dim = kwargs.get("u_dim", 2)
    x_dim = kwargs.get("x_dim", 4)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples")

    # Define all methods to evaluate
    methods = ["nearest_neighbors", "jacobian", "cost_evolution", "barrier_function", "all"]
    
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
        init_ps_2d = jnp.array(sample["init_ps"])
        goals = jnp.array(sample["init_goals"])
        
        # Convert to 4D state (x, y, vx, vy)
        init_ps = jnp.array([jnp.array([init_ps_2d[i][0], init_ps_2d[i][1], 0.0, 0.0]) for i in range(n_agents)])
        
        # Create agents
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
            agent.create_loss_function_mask()
        
        # Create reference trajectories
        ref_trajs = jnp.array([jnp.linspace(init_ps[i][:2], goals[i], tsteps) for i in range(n_agents)])
        
        # Evaluate each method
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
                    collision_weight=collision_weight,
                    collision_scale=collision_scale,
                )
                
                # Compute metrics
                metrics = compute_metrics(
                    final_x_trajs, 
                    control_trajs_result, 
                    simulation_masks, 
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
    model_path = "log/gnn_full_MP_3_edge-metric_full_top-k_5/train_n_agents_10_T_50_obs_10_lr_0.001_bs_32_sigma1_0.75_sigma2_0.75_epochs_50_loss_type_ego_agent_cost/20251105_222834/psn_best_model.pkl"
    model_type = "gnn"  
    dataset_path = "src/data/eval_data_upto_10p"

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
        "u_dim": 2,
        "x_dim": 4,
    }

    eval_model(**args)
