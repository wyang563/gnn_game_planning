import jax
import jax.numpy as jnp
import sys
import os
# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import flax.linen as nn
from solver.point_agent import PointAgent
from solver.drone_agent import DroneAgent
from solver.solve import create_batched_loss_functions_mask 
from models.policies import nearest_neighbors_top_k, jacobian_top_k, barrier_function_top_k, cost_evolution_top_k
from typing import Any, Dict, List
import json
from pathlib import Path
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

def eval_model(
    **kwargs: Any
) -> None:

    # new model parameters
    model_path = kwargs["model_path"]
    model = kwargs["model"]
    model_state = kwargs["model_state"]
    agent_type = kwargs["agent_type"]

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
    
    # Add the provided model to methods if it exists
    if model is not None and model_state is not None:
        methods.append("model_gnn")
    
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
