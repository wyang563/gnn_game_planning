import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import zarr
import os
import argparse
from data.mlp_dataset import create_mlp_dataloader

def load_single_data_instance(dataset_dir: str):
    """
    Load a single instance of data from the dataset.
    
    Args:
        dataset_dir: Path to the dataset directory containing .zarr files
        
    Returns:
        Tuple of (inputs, x0s, ref_trajs, targets) for a single timestep
    """
    # Create dataloader
    dataloader = create_mlp_dataloader(
        inputs_path=os.path.join(dataset_dir, "inputs.zarr"),
        targets_path=os.path.join(dataset_dir, "targets.zarr"),
        x0s_path=os.path.join(dataset_dir, "x0s.zarr"),
        ref_trajs_path=os.path.join(dataset_dir, "ref_trajs.zarr"),
        shuffle_buffer=500,
    )
    
    # Get first instance
    batch_iter = dataloader.create_jax_iterator()
    inputs, x0s, ref_trajs, targets = next(batch_iter)
    
    # Convert to numpy for plotting
    inputs = np.array(inputs)
    x0s = np.array(x0s)
    ref_trajs = np.array(ref_trajs)
    targets = np.array(targets)
    
    return inputs, x0s, ref_trajs, targets

def plot_agent_trajectories(inputs, x0s, ref_trajs, targets, n_agents):
    """
    Plot the past trajectories, current positions, reference trajectories, and target trajectories for all agents.
    
    Args:
        inputs: Input features (batch, n_agents, n_agents, horizon, state_dim) - past trajectories in inputs[0, :, :, :2]
        x0s: Initial states (batch, N, 4) - [x, y, vx, vy]
        ref_trajs: Reference trajectories (batch, N, horizon, 2) - [x, y]
        targets: Target trajectories (batch, N, horizon, 4) - [x, y, vx, vy]
        n_agents: Number of agents
        horizon: Planning horizon
    """
    # Remove batch dimension
    inputs = inputs.squeeze(0)  # (n_agents, n_agents, horizon, state_dim)
    x0s = x0s.squeeze(0)  # (n_agents, 4)
    ref_trajs = ref_trajs.squeeze(0)  # (n_agents, horizon, 2)
    targets = targets.squeeze(0)  # (n_agents, horizon, 4)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Extract past trajectories from inputs
    # inputs[0, :, :, :2] gives past positions for each agent
    past_trajs = inputs[0, :, :, :2]  # (n_agents, horizon, 2)
    
    # Extract current positions
    x0_positions = x0s[:, :2]  # (n_agents, 2)
    
    # Extract target trajectories (first two indices in last dimension)
    target_trajs = targets[:, :, :2]  # (n_agents, horizon, 2)
    
    # Plot for each agent
    for i in range(n_agents):
        color = f'C{i}'
        
        # 1) Plot past trajectories (dotted line)
        past_x = past_trajs[i, :, 0]
        past_y = past_trajs[i, :, 1]
        ax.plot(past_x, past_y, ':', alpha=0.7, linewidth=2, color=color, 
                label=f'Agent {i} Past' if i < 6 else "")
        
        # 2) Plot current positions (x0)
        ax.scatter(x0_positions[i, 0], x0_positions[i, 1], c=color, s=100, 
                  marker='o', edgecolors='black', linewidth=2, zorder=10,
                  label=f'Agent {i} Current' if i < 6 else "")
        
        # 3) Plot reference trajectories (solid line)
        ref_x = ref_trajs[i, :, 0]
        ref_y = ref_trajs[i, :, 1]
        ax.plot(ref_x, ref_y, '-', alpha=0.8, linewidth=2, color=color,
                label=f'Agent {i} Reference' if i < 6 else "")
        
        # 4) Plot target trajectories (dashed line)
        target_x = target_trajs[i, :, 0]
        target_y = target_trajs[i, :, 1]
        ax.plot(target_x, target_y, '--', alpha=0.8, linewidth=2, color=color,
                label=f'Agent {i} Target' if i < 6 else "")
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Agent Trajectories: Past (dotted), Current (circles), Reference (solid), Target (dashed)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    return fig

def main(save_plots=True, output_dir="plots", dataset_dir="src/data/mlp_n_agents_10_test"):
    """
    Main function to load data and create all plots.
    
    Args:
        save_plots: If True, save plots to files instead of displaying
        output_dir: Directory to save plots (only used if save_plots=True)
        dataset_dir: Path to the dataset directory containing .zarr files
    """
    
    print("Loading data...")
    try:
        inputs, x0s, ref_trajs, targets = load_single_data_instance(dataset_dir)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Remove batch dimension to get correct shapes
    targets_no_batch = targets.squeeze(0)
    n_agents, horizon = targets_no_batch.shape[:2]
    
    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to {output_dir}/")
    
    # Create the single plot
    print("Creating plot...")
    
    # Single plot: Past trajectories, current positions, reference trajectories, and target trajectories
    fig = plot_agent_trajectories(inputs, x0s, ref_trajs, targets, n_agents)
    
    if save_plots:
        fig.savefig(f"{output_dir}/agent_trajectories.png", dpi=300, bbox_inches='tight')
        print("Saved: agent_trajectories.png")
    
    if not save_plots:
        # Show the plot
        plt.show()
    else:
        # Close figure to free memory
        plt.close(fig)
    
    print("All plots created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from the GNN game planning dataset')
    parser.add_argument('--dataset_dir', type=str, default='src/data/mlp_n_agents_10_test',
                        help='Path to the dataset directory containing .zarr files')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files instead of displaying')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots (only used with --save)')
    
    args = parser.parse_args()
    
    # Call main function with command line arguments
    main(save_plots=args.save, output_dir=args.output_dir, dataset_dir=args.dataset_dir)
