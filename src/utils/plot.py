import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional
from PIL import Image


def plot_trajs(trajs, goals, init_points, ax=None, title: Optional[str] = None, 
               show_legend: bool = True, save_path: Optional[str] = None):
    """
    Plot trajectories for multiple agents with goals and initial positions.
    
    Args:
        trajs: Array of shape (n_agents, n_timesteps, 2) or (n_agents, n_timesteps, 4)
               containing the trajectories (x, y positions, possibly with velocities)
        goals: Array of shape (n_agents, 2) containing goal positions for each agent
        init_points: Array of shape (n_agents, 2) containing initial positions for each agent
        ax: Optional matplotlib axis to plot on. If None, creates a new figure
        title: Optional title for the plot
        show_legend: Whether to show the legend (default: True)
        save_path: Optional file path to save the plot. If provided, saves the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Convert to numpy arrays if needed
    trajs = np.array(trajs)
    goals = np.array(goals)
    init_points = np.array(init_points)
    
    # Extract position data (first 2 columns if state includes velocity)
    if trajs.shape[-1] > 2:
        positions = trajs[:, :, :2]
    else:
        positions = trajs
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()
    
    n_agents = positions.shape[0]
    
    # Get colors for each agent using tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    # Plot each agent's trajectory
    for i in range(n_agents):
        color = colors[i]
        
        # Plot trajectory
        ax.plot(positions[i, :, 0], positions[i, :, 1], 
                color=color, linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Plot initial position (circle)
        ax.scatter(init_points[i, 0], init_points[i, 1], 
                   color=color, s=120, marker='o', edgecolors='black', 
                   linewidth=2, zorder=5)
        
        # Plot goal (star)
        ax.scatter(goals[i, 0], goals[i, 1], 
                   color=color, s=200, marker='*', edgecolors='black',
                   linewidth=2, zorder=5)
    
    # Set plot properties
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Agent Trajectories (N={n_agents})', fontsize=14)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_agent_gif(trajectories, goals, init_positions, simulation_masks, ego_agent_id, 
                   save_path, fps=10, figsize=(12, 10), xlim=(-3.5, 3.5), ylim=(-3.5, 3.5)):
    """
    Create a GIF animation showing agent trajectories with mask-based highlighting.
    
    Args:
        trajectories: Array of shape (n_agents, n_timesteps, state_dim) containing trajectories
        goals: Array of shape (n_agents, 2) containing goal positions for each agent
        init_positions: Array of shape (n_agents, 2) containing initial positions
        simulation_masks: Array of shape (n_timesteps, n_agents) or (n_agents, n_timesteps) 
                         boolean mask indicating which agents are selected at each timestep
        ego_agent_id: int, ID of the ego agent to highlight in blue
        save_path: str, path to save the GIF file
        fps: int, frames per second for the GIF (default: 10)
        figsize: tuple, figure size (default: (12, 10))
        xlim: tuple, x-axis limits (default: (-3.5, 3.5))
        ylim: tuple, y-axis limits (default: (-3.5, 3.5))
    """
    # Convert to numpy arrays
    trajectories = np.array(trajectories)
    goals = np.array(goals)
    init_positions = np.array(init_positions)
    simulation_masks = np.array(simulation_masks)
    
    # Extract position data (first 2 columns if state includes velocity)
    if trajectories.shape[-1] > 2:
        positions = trajectories[:, :, :2]
    else:
        positions = trajectories
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Handle mask shape - convert to (n_timesteps, n_agents)
    if simulation_masks.shape[0] == n_agents and simulation_masks.shape[1] == n_timesteps:
        simulation_masks = simulation_masks.T
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    
    print(f"Creating GIF with {n_timesteps} frames...")
    
    # Generate all frames
    frames = []
    for step in range(n_timesteps):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Title
        ax.set_title(f'Step {step+1}/{n_timesteps}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Get selected agents at this timestep
        selected_agents = np.where(simulation_masks[step])[0] if step < len(simulation_masks) else []
        
        # Plot trajectories for all agents (gradually accumulated up to current step)
        for i in range(n_agents):
            # Only plot trajectory up to current step
            traj = positions[i, :step+1, :]
            
            is_selected = i in selected_agents
            
            if i == ego_agent_id:  # Ego agent
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], '-', 
                           color=ego_color, alpha=0.9, linewidth=3, 
                           label=f'Ego Agent {i}')
            else:  # Other agents
                if len(traj) > 1:
                    if is_selected:
                        # Selected agent: solid line, red color
                        ax.plot(traj[:, 0], traj[:, 1], '-', 
                               color=selected_color, alpha=0.8, linewidth=2, 
                               label=f'Agent {i} (Selected)')
                    else:
                        # Non-selected agent: dashed line, gray color
                        ax.plot(traj[:, 0], traj[:, 1], '--', 
                               color=other_agent_color, alpha=0.4, linewidth=1, 
                               label=f'Agent {i}')
        
        # Plot current positions
        for i in range(n_agents):
            is_selected = i in selected_agents
            current_pos = positions[i, step, :]
            
            if i == ego_agent_id:
                # Ego agent: blue circle
                ax.plot(current_pos[0], current_pos[1], 'o', 
                       color=ego_color, markersize=10, alpha=0.8)
            else:
                # Other agents: colored by selection status
                if is_selected:
                    ax.plot(current_pos[0], current_pos[1], 'o', 
                           color=selected_color, markersize=8, alpha=0.8)
                    # Add text label with selection indicator
                    ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}*', 
                           fontsize=10, ha='left', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                else:
                    ax.plot(current_pos[0], current_pos[1], 'o', 
                           color=other_agent_color, markersize=6, alpha=0.6)
                    # Add text label
                    ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}', 
                           fontsize=10, ha='left', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Plot goals as stars
        for i in range(n_agents):
            goal_pos = goals[i]
            
            if i == ego_agent_id:  # Ego agent goal
                ax.scatter(goal_pos[0], goal_pos[1], 
                          color=ego_color, s=200, marker='*', 
                          edgecolors='black', linewidth=2, alpha=0.8)
            else:  # Other agent goals
                ax.scatter(goal_pos[0], goal_pos[1], 
                          color=other_agent_color, s=150, marker='*', 
                          edgecolors='black', linewidth=1.5, alpha=0.6)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(buf, 'RGBA')
        
        frames.append(img)
        plt.close(fig)
        
        if (step + 1) % 10 == 0 or step == n_timesteps - 1:
            print(f"  Generated frame {step + 1}/{n_timesteps}")
    
    # Save as GIF
    print(f"Saving GIF to: {save_path}")
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,  # Convert fps to duration in ms
        loop=0
    )
    print(f"✓ GIF created successfully!")

