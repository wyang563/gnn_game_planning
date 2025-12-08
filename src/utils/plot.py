import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from typing import Optional, Union
from PIL import Image
import jax.numpy as jnp

# ============================== Point Agent Plotting Functions ==============================

def plot_point_agent_trajs(trajs, goals, init_points, ax=None, title: Optional[str] = None, 
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
    
    # Get colors for each agent using a colormap that supports many colors
    # Use hsv colormap which cycles through colors and gives good distinction
    colors = plt.cm.hsv(np.linspace(0, 1, n_agents))
    
    # Plot each agent's trajectory
    for i in range(n_agents):
        color = colors[i]
        
        # Plot trajectory
        ax.plot(positions[i, :, 0], positions[i, :, 1], 
                color=color, linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Plot initial position (circle) with label
        start_label = f'Start {i}' if i == 0 else ''
        ax.scatter(init_points[i, 0], init_points[i, 1], 
                   color=color, s=120, marker='o', edgecolors='black', 
                   linewidth=2, zorder=5, label=start_label)
        
        # Add text label for start point
        ax.text(init_points[i, 0], init_points[i, 1], f' {i}', 
                fontsize=9, ha='left', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5),
                zorder=6)
        
        # Plot goal (star) with label
        goal_label = f'Goal {i}' if i == 0 else ''
        ax.scatter(goals[i, 0], goals[i, 1], 
                   color=color, s=200, marker='*', edgecolors='black',
                   linewidth=2, zorder=5, label=goal_label)
        
        # Add text label for goal point
        ax.text(goals[i, 0], goals[i, 1], f' {i}', 
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5),
                zorder=6)
    
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


def plot_point_agent_trajs_gif(trajs, goals, init_points, save_path: str, 
                               fps: int = 10, figsize: tuple = (12, 10), 
                               title: Optional[str] = None):
    """
    Create a GIF animation showing point agent trajectories gradually traced out over time.
    
    Args:
        trajs: Array of shape (n_agents, n_timesteps, 2) or (n_agents, n_timesteps, 4+)
               containing the trajectories (x, y positions, possibly with velocities)
        goals: Array of shape (n_agents, 2) containing goal positions for each agent
        init_points: Array of shape (n_agents, 2) containing initial positions for each agent
        save_path: str, path to save the GIF file (required)
        fps: int, frames per second for the GIF (default: 10)
        figsize: tuple, figure size (default: (12, 10))
        title: Optional title for the plot
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
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Auto-calculate axis limits
    all_x = np.concatenate([
        positions[:, :, 0].flatten(),
        goals[:, 0],
        init_points[:, 0]
    ])
    all_y = np.concatenate([
        positions[:, :, 1].flatten(),
        goals[:, 1],
        init_points[:, 1]
    ])
    
    # Calculate ranges with padding (15% of the range)
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_padding = max(0.15 * x_range, 0.5)
    y_padding = max(0.15 * y_range, 0.5)
    
    xlim = (x_min - x_padding, x_max + x_padding)
    ylim = (y_min - y_padding, y_max + y_padding)
    
    # Get colors for each agent using hsv colormap
    colors = plt.cm.hsv(np.linspace(0, 1, n_agents))
    
    print(f"Creating GIF with {n_timesteps} frames...")
    
    # Generate all frames
    frames = []
    for step in range(n_timesteps):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        
        # Title
        if title:
            ax.set_title(f'{title} - Step {step+1}/{n_timesteps}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Point Agent Trajectories - Step {step+1}/{n_timesteps}', 
                        fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        
        # Plot trajectories for all agents (gradually accumulated up to current step)
        for i in range(n_agents):
            color = colors[i]
            # Only plot trajectory up to current step
            traj = positions[i, :step+1, :]
            
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1],
                       color=color, linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Plot current positions
        for i in range(n_agents):
            color = colors[i]
            current_pos = positions[i, step, :]
            
            # Plot current position as a circle
            ax.scatter(current_pos[0], current_pos[1],
                      color=color, s=100, marker='o', edgecolors='black',
                      linewidth=2, alpha=0.9, zorder=5)
        
        # Plot start points (initial positions)
        for i in range(n_agents):
            color = colors[i]
            ax.scatter(init_points[i, 0], init_points[i, 1],
                      color=color, s=120, marker='o', edgecolors='black',
                      linewidth=2, alpha=0.8, zorder=5)
            
            # Add text label for start point
            ax.text(init_points[i, 0], init_points[i, 1], f' {i}',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, 
                            edgecolor='black', linewidth=0.5))
        
        # Plot goal points
        for i in range(n_agents):
            color = colors[i]
            ax.scatter(goals[i, 0], goals[i, 1],
                      color=color, s=200, marker='*', edgecolors='black',
                      linewidth=2, alpha=0.8, zorder=5)
            
            # Add text label for goal point
            ax.text(goals[i, 0], goals[i, 1], f' {i}',
                   fontsize=9, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, 
                            edgecolor='black', linewidth=0.5))
        
        ax.grid(True, alpha=0.3)
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

def plot_point_agent_gif(trajectories, goals, init_positions, simulation_masks, ego_agent_id, 
                   save_path, fps=10, figsize=(12, 10), xlim=None, ylim=None):
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
        xlim: tuple, x-axis limits (default: None, auto-calculated from data)
        ylim: tuple, y-axis limits (default: None, auto-calculated from data)
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
    
    # Auto-calculate axis limits if not provided
    if xlim is None or ylim is None:
        # Gather all x and y coordinates from trajectories, goals, and initial positions
        all_x = np.concatenate([
            positions[:, :, 0].flatten(),
            goals[:, 0],
            init_positions[:, 0]
        ])
        all_y = np.concatenate([
            positions[:, :, 1].flatten(),
            goals[:, 1],
            init_positions[:, 1]
        ])
        
        # Calculate min and max with padding (15% of the range)
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_padding = max(0.15 * x_range, 0.5)  # At least 0.5 units of padding
        y_padding = max(0.15 * y_range, 0.5)
        
        if xlim is None:
            xlim = (x_min - x_padding, x_max + x_padding)
        if ylim is None:
            ylim = (y_min - y_padding, y_max + y_padding)
    
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


def plot_past_and_predicted_point_agent_trajectories(x_trajs, dt: float, model=None, 
                                         ax=None, title: Optional[str] = None,
                                         show_legend: bool = True, 
                                         save_path: Optional[str] = None):
    """
    Plot past trajectories along with predicted future trajectories from GNNSelectionNetwork.
    
    This function is designed to test the predict_future_trajectory method of GNNSelectionNetwork.
    It plots:
    - Past trajectories (solid lines) from x_trajs
    - Predicted future trajectories (dashed lines) computed using predict_future_trajectory
    
    Args:
        x_trajs: Past trajectory data
            - Shape (T_observation, N_agents, 4) for single trajectory
            - Shape (batch_size, T_observation, N_agents, 4) for batch (only first item plotted)
            Contains states [x, y, vx, vy] for each agent at each timestep
        dt: Time step size (delta time) used for trajectory prediction
        model: Optional GNNSelectionNetwork model instance. If provided, uses model.predict_future_trajectory.
               If None, implements the prediction logic directly.
        ax: Optional matplotlib axis to plot on. If None, creates a new figure
        title: Optional title for the plot
        show_legend: Whether to show the legend (default: True)
        save_path: Optional file path to save the plot. If provided, saves the figure
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Convert to numpy/jax array if needed
    if isinstance(x_trajs, np.ndarray):
        x_trajs_jax = jnp.array(x_trajs)
    else:
        x_trajs_jax = x_trajs
    
    # Handle batch dimension - if batch, take first item
    if len(x_trajs_jax.shape) == 4:  # (batch_size, T_obs, N_agents, 4)
        x_trajs_jax = x_trajs_jax[0]  # Take first batch item
    
    # Ensure shape is (T_observation, N_agents, 4)
    x_trajs_jax = jnp.atleast_3d(x_trajs_jax)
    if x_trajs_jax.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4 (x, y, vx, vy), got {x_trajs_jax.shape[-1]}")
    
    # Add batch dimension for prediction: (1, T_obs, N_agents, 4)
    x_trajs_batch = jnp.expand_dims(x_trajs_jax, axis=0)
    
    # Predict future trajectories
    if model is not None:
        # Use model's predict_future_trajectory method
        predicted_trajs_batch = model.predict_future_trajectory(x_trajs_batch)
    else:
        # Implement prediction logic directly (same as in GNNSelectionNetwork)
        T_obs = x_trajs_batch.shape[1]
        
        # Extract last known state (most recent observation)
        last_positions = x_trajs_batch[:, -1, :, :2]  # (batch, N_agents, 2)
        last_velocities = x_trajs_batch[:, -1, :, 2:]  # (batch, N_agents, 2)
        
        # Estimate acceleration using finite differences from recent velocity changes
        if T_obs >= 2:
            velocity_diff = x_trajs_batch[:, -1, :, 2:] - x_trajs_batch[:, -2, :, 2:]
            acceleration = velocity_diff / dt
        else:
            acceleration = jnp.zeros_like(last_velocities)
        
        # Predict future trajectories using constant acceleration model
        future_trajectories = []
        
        for t_future in range(1, T_obs + 1):
            # Time from last observation
            delta_t = t_future * dt
            
            # Position: p(t) = p0 + v0*t + 0.5*a*t^2
            future_positions = (last_positions + last_velocities * delta_t + 
                               0.5 * acceleration * (delta_t ** 2))
            
            # Velocity: v(t) = v0 + a*t
            future_velocities = last_velocities + acceleration * delta_t
            
            # Concatenate position and velocity
            future_state = jnp.concatenate([future_positions, future_velocities], axis=-1)
            future_trajectories.append(future_state)
        
        # Stack along time dimension: (batch_size, T_observation, N_agents, 4)
        predicted_trajs_batch = jnp.stack(future_trajectories, axis=1)
    
    # Remove batch dimension and convert to numpy
    predicted_trajs = np.array(predicted_trajs_batch[0])  # (T_obs, N_agents, 4)
    past_trajs = np.array(x_trajs_jax)  # (T_obs, N_agents, 4)
    
    # Extract position data (first 2 columns)
    past_positions = past_trajs[:, :, :2]  # (T_obs, N_agents, 2)
    predicted_positions = predicted_trajs[:, :, :2]  # (T_obs, N_agents, 2)
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    T_obs, n_agents, _ = past_positions.shape
    
    # Get colors for each agent using tab10 colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    # Plot past trajectories (solid lines)
    for i in range(n_agents):
        color = colors[i]
        ax.plot(past_positions[:, i, 0], past_positions[:, i, 1],
                color=color, linewidth=2.5, alpha=0.8, linestyle='-',
                label=f'Agent {i} (Past)', zorder=3)
    
    # Plot predicted future trajectories (dashed lines)
    # Connect past to future at the transition point
    for i in range(n_agents):
        color = colors[i]
        # Get the last past position
        last_past_pos = past_positions[-1, i, :]
        # Get the first predicted position
        first_pred_pos = predicted_positions[0, i, :]
        
        # Plot predicted trajectory (dashed)
        ax.plot(predicted_positions[:, i, 0], predicted_positions[:, i, 1],
                color=color, linewidth=2.5, alpha=0.6, linestyle='--',
                label=f'Agent {i} (Predicted)', zorder=2)
        
        # Draw a connecting line from past to future (thin, lighter)
        ax.plot([last_past_pos[0], first_pred_pos[0]], 
                [last_past_pos[1], first_pred_pos[1]],
                color=color, linewidth=1, alpha=0.4, linestyle=':', zorder=1)
    
    # Mark the transition point (where past ends and future begins)
    for i in range(n_agents):
        color = colors[i]
        transition_pos = past_positions[-1, i, :]
        ax.scatter(transition_pos[0], transition_pos[1],
                  color=color, s=100, marker='o', edgecolors='black',
                  linewidth=2, zorder=5, label=f'Transition (Agent {i})' if i == 0 else '')
    
    # Set plot properties
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Past vs Predicted Future Trajectories (N={n_agents})', 
                    fontsize=14, fontweight='bold')
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text annotation to clarify the plot
    ax.text(0.02, 0.98, 'Solid: Past trajectories\nDashed: Predicted future',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Only close if we created the figure (ax was None)
    if ax is None:
        plt.close()
    
    return fig, ax

# ============================== Drone Agent Plotting Functions ==============================

def plot_drone_agent_trajs(trajs, goals, init_points, ax=None, title: Optional[str] = None, 
               show_legend: bool = True, save_path: Optional[str] = None):
    """
    Plot trajectories for multiple drone agents with goals and initial positions in 3D.
    
    Args:
        trajs: Array of shape (n_agents, n_timesteps, 3) or (n_agents, n_timesteps, 6+)
               containing the trajectories (x, y, z positions, possibly with velocities)
        goals: Array of shape (n_agents, 3) containing goal positions for each agent
        init_points: Array of shape (n_agents, 3) containing initial positions for each agent
        ax: Optional matplotlib 3D axis to plot on. If None, creates a new figure with 3D projection
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
    
    # Extract position data (first 3 columns if state includes velocity)
    if trajs.shape[-1] > 3:
        positions = trajs[:, :, :3]
    else:
        positions = trajs
    
    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    
    n_agents = positions.shape[0]
    
    # Get colors for each agent using a colormap that supports many colors
    # Use hsv colormap which cycles through colors and gives good distinction
    colors = plt.cm.hsv(np.linspace(0, 1, n_agents))
    
    # Plot each agent's trajectory
    for i in range(n_agents):
        color = colors[i]
        
        # Plot trajectory
        ax.plot(positions[i, :, 0], positions[i, :, 1], positions[i, :, 2],
                color=color, linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Plot initial position (circle) with label
        start_label = f'Start {i}' if i == 0 else ''
        ax.scatter(init_points[i, 0], init_points[i, 1], init_points[i, 2],
                   color=color, s=120, marker='o', edgecolors='black', 
                   linewidth=2, zorder=5, label=start_label)
        
        # Add text label for start point
        ax.text(init_points[i, 0], init_points[i, 1], init_points[i, 2], f' {i}', 
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5))
        
        # Plot goal (star) with label
        goal_label = f'Goal {i}' if i == 0 else ''
        ax.scatter(goals[i, 0], goals[i, 1], goals[i, 2],
                   color=color, s=200, marker='*', edgecolors='black',
                   linewidth=2, zorder=5, label=goal_label)
        
        # Add text label for goal point
        ax.text(goals[i, 0], goals[i, 1], goals[i, 2], f' {i}', 
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5))
    
    # Set plot properties
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Drone Agent Trajectories (N={n_agents})', fontsize=14)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set equal aspect ratio for 3D plot
    # Get the limits for each axis
    all_x = np.concatenate([
        positions[:, :, 0].flatten(),
        goals[:, 0],
        init_points[:, 0]
    ])
    all_y = np.concatenate([
        positions[:, :, 1].flatten(),
        goals[:, 1],
        init_points[:, 1]
    ])
    all_z = np.concatenate([
        positions[:, :, 2].flatten(),
        goals[:, 2],
        init_points[:, 2]
    ])
    
    # Calculate ranges
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range)
    
    # Center the plot
    x_center = (all_x.max() + all_x.min()) / 2
    y_center = (all_y.max() + all_y.min()) / 2
    z_center = (all_z.max() + all_z.min()) / 2
    
    # Set limits with equal aspect ratio
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_drone_agent_trajs_gif(trajs, goals, init_points, save_path: str, 
                               fps: int = 10, figsize: tuple = (12, 10), 
                               title: Optional[str] = None):
    """
    Create a GIF animation showing drone agent trajectories gradually traced out over time.
    
    Args:
        trajs: Array of shape (n_agents, n_timesteps, 3) or (n_agents, n_timesteps, 6+)
               containing the trajectories (x, y, z positions, possibly with velocities)
        goals: Array of shape (n_agents, 3) containing goal positions for each agent
        init_points: Array of shape (n_agents, 3) containing initial positions for each agent
        save_path: str, path to save the GIF file (required)
        fps: int, frames per second for the GIF (default: 10)
        figsize: tuple, figure size (default: (12, 10))
        title: Optional title for the plot
    """
    # Convert to numpy arrays if needed
    trajs = np.array(trajs)
    goals = np.array(goals)
    init_points = np.array(init_points)
    
    # Extract position data (first 3 columns if state includes velocity)
    if trajs.shape[-1] > 3:
        positions = trajs[:, :, :3]
    else:
        positions = trajs
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Auto-calculate axis limits
    all_x = np.concatenate([
        positions[:, :, 0].flatten(),
        goals[:, 0],
        init_points[:, 0]
    ])
    all_y = np.concatenate([
        positions[:, :, 1].flatten(),
        goals[:, 1],
        init_points[:, 1]
    ])
    all_z = np.concatenate([
        positions[:, :, 2].flatten(),
        goals[:, 2],
        init_points[:, 2]
    ])
    
    # Calculate ranges with padding (15% of the range)
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_padding = max(0.15 * x_range, 0.5)
    y_padding = max(0.15 * y_range, 0.5)
    z_padding = max(0.15 * z_range, 0.5)
    
    xlim = (x_min - x_padding, x_max + x_padding)
    ylim = (y_min - y_padding, y_max + y_padding)
    zlim = (z_min - z_padding, z_max + z_padding)
    
    # Get colors for each agent using hsv colormap
    colors = plt.cm.hsv(np.linspace(0, 1, n_agents))
    
    print(f"Creating GIF with {n_timesteps} frames...")
    
    # Generate all frames
    frames = []
    for step in range(n_timesteps):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        # Title
        if title:
            ax.set_title(f'{title} - Step {step+1}/{n_timesteps}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Drone Agent Trajectories - Step {step+1}/{n_timesteps}', 
                        fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_zlabel('Z Position (m)', fontsize=10)
        
        # Plot trajectories for all agents (gradually accumulated up to current step)
        for i in range(n_agents):
            color = colors[i]
            # Only plot trajectory up to current step
            traj = positions[i, :step+1, :]
            
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=color, linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Plot current positions
        for i in range(n_agents):
            color = colors[i]
            current_pos = positions[i, step, :]
            
            # Plot current position as a circle
            ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                      color=color, s=100, marker='o', edgecolors='black',
                      linewidth=2, alpha=0.9, zorder=5)
        
        # Plot start points (initial positions)
        for i in range(n_agents):
            color = colors[i]
            ax.scatter(init_points[i, 0], init_points[i, 1], init_points[i, 2],
                      color=color, s=120, marker='o', edgecolors='black',
                      linewidth=2, alpha=0.8, zorder=5)
            
            # Add text label for start point
            ax.text(init_points[i, 0], init_points[i, 1], init_points[i, 2], f' {i}',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, 
                            edgecolor='black', linewidth=0.5))
        
        # Plot goal points
        for i in range(n_agents):
            color = colors[i]
            ax.scatter(goals[i, 0], goals[i, 1], goals[i, 2],
                      color=color, s=200, marker='*', edgecolors='black',
                      linewidth=2, alpha=0.8, zorder=5)
            
            # Add text label for goal point
            ax.text(goals[i, 0], goals[i, 1], goals[i, 2], f' {i}',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, 
                            edgecolor='black', linewidth=0.5))
        
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

def plot_drone_agent_gif(trajectories, goals, init_positions, simulation_masks, ego_agent_id, 
                   save_path, fps=10, figsize=(12, 10), xlim=None, ylim=None, zlim=None):
    """
    Create a GIF animation showing drone agent trajectories with mask-based highlighting in 3D.
    
    Args:
        trajectories: Array of shape (n_agents, n_timesteps, state_dim) containing trajectories
        goals: Array of shape (n_agents, 3) containing goal positions for each agent
        init_positions: Array of shape (n_agents, 3) containing initial positions
        simulation_masks: Array of shape (n_timesteps, n_agents) or (n_agents, n_timesteps) 
                         boolean mask indicating which agents are selected at each timestep
        ego_agent_id: int, ID of the ego agent to highlight in blue
        save_path: str, path to save the GIF file
        fps: int, frames per second for the GIF (default: 10)
        figsize: tuple, figure size (default: (12, 10))
        xlim: tuple, x-axis limits (default: None, auto-calculated from data)
        ylim: tuple, y-axis limits (default: None, auto-calculated from data)
        zlim: tuple, z-axis limits (default: None, auto-calculated from data)
    """
    # Convert to numpy arrays
    trajectories = np.array(trajectories)
    goals = np.array(goals)
    init_positions = np.array(init_positions)
    simulation_masks = np.array(simulation_masks)
    
    # Extract position data (first 3 columns if state includes velocity)
    if trajectories.shape[-1] > 3:
        positions = trajectories[:, :, :3]
    else:
        positions = trajectories
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Handle mask shape - convert to (n_timesteps, n_agents)
    if simulation_masks.shape[0] == n_agents and simulation_masks.shape[1] == n_timesteps:
        simulation_masks = simulation_masks.T
    
    # Auto-calculate axis limits if not provided
    if xlim is None or ylim is None or zlim is None:
        # Gather all x, y, z coordinates from trajectories, goals, and initial positions
        all_x = np.concatenate([
            positions[:, :, 0].flatten(),
            goals[:, 0],
            init_positions[:, 0]
        ])
        all_y = np.concatenate([
            positions[:, :, 1].flatten(),
            goals[:, 1],
            init_positions[:, 1]
        ])
        all_z = np.concatenate([
            positions[:, :, 2].flatten(),
            goals[:, 2],
            init_positions[:, 2]
        ])
        
        # Calculate min and max with padding (15% of the range)
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_padding = max(0.15 * x_range, 0.5)  # At least 0.5 units of padding
        y_padding = max(0.15 * y_range, 0.5)
        z_padding = max(0.15 * z_range, 0.5)
        
        if xlim is None:
            xlim = (x_min - x_padding, x_max + x_padding)
        if ylim is None:
            ylim = (y_min - y_padding, y_max + y_padding)
        if zlim is None:
            zlim = (z_min - z_padding, z_max + z_padding)
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    
    print(f"Creating GIF with {n_timesteps} frames...")
    
    # Generate all frames
    frames = []
    for step in range(n_timesteps):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        # Title
        ax.set_title(f'Step {step+1}/{n_timesteps}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
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
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', 
                           color=ego_color, alpha=0.9, linewidth=3, 
                           label=f'Ego Agent {i}')
            else:  # Other agents
                if len(traj) > 1:
                    if is_selected:
                        # Selected agent: solid line, red color
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', 
                               color=selected_color, alpha=0.8, linewidth=2, 
                               label=f'Agent {i} (Selected)')
                    else:
                        # Non-selected agent: dashed line, gray color
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '--', 
                               color=other_agent_color, alpha=0.4, linewidth=1, 
                               label=f'Agent {i}')
        
        # Plot current positions
        for i in range(n_agents):
            is_selected = i in selected_agents
            current_pos = positions[i, step, :]
            
            if i == ego_agent_id:
                # Ego agent: blue circle
                ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                          color=ego_color, s=100, marker='o', alpha=0.8,
                          edgecolors='black', linewidth=2)
            else:
                # Other agents: colored by selection status
                if is_selected:
                    ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                              color=selected_color, s=80, marker='o', alpha=0.8,
                              edgecolors='black', linewidth=1.5)
                    # Add text label with selection indicator
                    ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, current_pos[2] + 0.1, 
                           f'{i}*', fontsize=10, ha='left', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                else:
                    ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                              color=other_agent_color, s=60, marker='o', alpha=0.6,
                              edgecolors='black', linewidth=1)
                    # Add text label
                    ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, current_pos[2] + 0.1, 
                           f'{i}', fontsize=10, ha='left', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Plot goals as stars
        for i in range(n_agents):
            goal_pos = goals[i]
            
            if i == ego_agent_id:  # Ego agent goal
                ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                          color=ego_color, s=200, marker='*', 
                          edgecolors='black', linewidth=2, alpha=0.8)
            else:  # Other agent goals
                ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
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

def plot_drone_agent_mask_png(trajectories, goals, init_positions, player_mask, ego_agent_id, 
                              save_path, figsize=(12, 10), xlim=None, ylim=None, zlim=None):
    """
    Create a single PNG plot showing drone agent trajectories with mask-based coloring.
    Ego agent's trajectory is always blue. Other agents' trajectories are red when included
    in the ego agent's player mask, gray otherwise.
    
    Args:
        trajectories: Array of shape (n_agents, n_timesteps, state_dim) containing trajectories
        goals: Array of shape (n_agents, 3) containing goal positions for each agent
        init_positions: Array of shape (n_agents, 3) containing initial positions
        player_mask: Array of shape (n_timesteps, n_agents) or (n_agents, n_timesteps) 
                   boolean mask indicating which agents are in ego agent's player mask at each timestep
        ego_agent_id: int, ID of the ego agent to highlight in blue
        save_path: str, path to save the PNG file
        figsize: tuple, figure size (default: (12, 10))
        xlim: tuple, x-axis limits (default: None, auto-calculated from data)
        ylim: tuple, y-axis limits (default: None, auto-calculated from data)
        zlim: tuple, z-axis limits (default: None, auto-calculated from data)
    """
    # Convert to numpy arrays
    trajectories = np.array(trajectories)
    goals = np.array(goals)
    init_positions = np.array(init_positions)
    player_mask = np.array(player_mask)
    
    # Extract position data (first 3 columns if state includes velocity)
    if trajectories.shape[-1] > 3:
        positions = trajectories[:, :, :3]
    else:
        positions = trajectories
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Handle mask shape - convert to (n_timesteps, n_agents)
    if player_mask.shape[0] == n_agents and player_mask.shape[1] == n_timesteps:
        player_mask = player_mask.T
    
    # Auto-calculate axis limits if not provided
    if xlim is None or ylim is None or zlim is None:
        # Gather all x, y, z coordinates from trajectories, goals, and initial positions
        all_x = np.concatenate([
            positions[:, :, 0].flatten(),
            goals[:, 0],
            init_positions[:, 0]
        ])
        all_y = np.concatenate([
            positions[:, :, 1].flatten(),
            goals[:, 1],
            init_positions[:, 1]
        ])
        all_z = np.concatenate([
            positions[:, :, 2].flatten(),
            goals[:, 2],
            init_positions[:, 2]
        ])
        
        # Calculate min and max with padding (15% of the range)
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_padding = max(0.15 * x_range, 0.5)  # At least 0.5 units of padding
        y_padding = max(0.15 * y_range, 0.5)
        z_padding = max(0.15 * z_range, 0.5)
        
        if xlim is None:
            xlim = (x_min - x_padding, x_max + x_padding)
        if ylim is None:
            ylim = (y_min - y_padding, y_max + y_padding)
        if zlim is None:
            zlim = (z_min - z_padding, z_max + z_padding)
    
    # Color scheme
    ego_color = 'darkblue'
    masked_color = 'red'
    unmasked_color = 'gray'
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    ax.set_title('Agent Trajectories with Player Mask', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.grid(True, alpha=0.3)
    
    # Plot ego agent's full trajectory in blue
    ego_traj = positions[ego_agent_id, :, :]
    if len(ego_traj) > 1:
        ax.plot(ego_traj[:, 0], ego_traj[:, 1], ego_traj[:, 2], '-', 
               color=ego_color, alpha=0.9, linewidth=3, 
               label=f'Ego Agent {ego_agent_id}')
    
    # Plot other agents' trajectories with mask-based coloring
    for i in range(n_agents):
        if i == ego_agent_id:
            continue  # Skip ego agent (already plotted)
        
        agent_traj = positions[i, :, :]
        if len(agent_traj) <= 1:
            continue
        
        # Get mask for this agent across all timesteps
        agent_mask = player_mask[:, i]  # Shape: (n_timesteps,)
        
        # Segment trajectory based on mask changes
        # Convert boolean to int and find where it changes
        mask_int = agent_mask.astype(int)
        diffs = np.diff(mask_int)
        # Find indices where mask changes (add 1 because diff gives index of second element)
        change_indices = np.where(diffs != 0)[0] + 1
        # Create segment boundaries: [0, change1, change2, ..., n_timesteps]
        segment_boundaries = np.concatenate([[0], change_indices, [n_timesteps]])
        
        # Plot each segment
        for seg_idx in range(len(segment_boundaries) - 1):
            start_idx = segment_boundaries[seg_idx]
            end_idx = segment_boundaries[seg_idx + 1]
            
            if start_idx >= end_idx:
                continue
            
            # Determine color based on mask value at start of segment
            is_masked = agent_mask[start_idx]
            
            # Extract segment
            seg_traj = agent_traj[start_idx:end_idx, :]
            
            if len(seg_traj) > 1:
                color = masked_color if is_masked else unmasked_color
                alpha = 0.8 if is_masked else 0.4
                linewidth = 2.5 if is_masked else 1.5
                linestyle = '-' if is_masked else '--'
                
                ax.plot(seg_traj[:, 0], seg_traj[:, 1], seg_traj[:, 2], 
                       linestyle, color=color, alpha=alpha, linewidth=linewidth,
                       label=f'Agent {i}' if seg_idx == 0 and is_masked else '')
    
    # Plot final positions
    for i in range(n_agents):
        final_pos = positions[i, -1, :]
        is_masked = player_mask[-1, i] if i < player_mask.shape[1] else False
        
        if i == ego_agent_id:
            # Ego agent: blue circle
            ax.scatter(final_pos[0], final_pos[1], final_pos[2],
                      color=ego_color, s=100, marker='o', alpha=0.8,
                      edgecolors='black', linewidth=2)
        else:
            # Other agents: colored by mask status
            color = masked_color if is_masked else unmasked_color
            ax.scatter(final_pos[0], final_pos[1], final_pos[2],
                      color=color, s=80, marker='o', alpha=0.8,
                      edgecolors='black', linewidth=1.5)
            # Add text label with mask indicator
            label_text = f'{i}*' if is_masked else f'{i}'
            ax.text(final_pos[0] + 0.1, final_pos[1] + 0.1, final_pos[2] + 0.1, 
                   label_text, fontsize=10, ha='left', va='bottom', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Plot goals as stars
    for i in range(n_agents):
        goal_pos = goals[i]
        
        if i == ego_agent_id:  # Ego agent goal
            ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                      color=ego_color, s=200, marker='*', 
                      edgecolors='black', linewidth=2, alpha=0.8)
        else:  # Other agent goals
            # Check if agent is ever in mask to determine goal color
            ever_masked = np.any(player_mask[:, i]) if i < player_mask.shape[1] else False
            goal_color = masked_color if ever_masked else unmasked_color
            ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                      color=goal_color, s=150, marker='*', 
                      edgecolors='black', linewidth=1.5, alpha=0.6)
    
    plt.tight_layout()
    
    # Save as PNG
    print(f"Saving PNG to: {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ PNG created successfully!")

def plot_point_agent_mask_png(trajectories, goals, init_positions, player_mask, ego_agent_id, 
                              save_path, figsize=(12, 10), xlim=None, ylim=None):
    """
    Create a single PNG plot showing point agent trajectories with mask-based coloring in 2D.
    Ego agent's trajectory is always blue. Other agents' trajectories are red when included
    in the ego agent's player mask, gray otherwise.
    
    Args:
        trajectories: Array of shape (n_agents, n_timesteps, state_dim) containing trajectories
        goals: Array of shape (n_agents, 2) containing goal positions for each agent
        init_positions: Array of shape (n_agents, 2) containing initial positions
        player_mask: Array of shape (n_timesteps, n_agents) or (n_agents, n_timesteps) 
                   boolean mask indicating which agents are in ego agent's player mask at each timestep
        ego_agent_id: int, ID of the ego agent to highlight in blue
        save_path: str, path to save the PNG file
        figsize: tuple, figure size (default: (12, 10))
        xlim: tuple, x-axis limits (default: None, auto-calculated from data)
        ylim: tuple, y-axis limits (default: None, auto-calculated from data)
    """
    # Convert to numpy arrays
    trajectories = np.array(trajectories)
    goals = np.array(goals)
    init_positions = np.array(init_positions)
    player_mask = np.array(player_mask)
    
    # Extract position data (first 2 columns if state includes velocity)
    if trajectories.shape[-1] > 2:
        positions = trajectories[:, :, :2]
    else:
        positions = trajectories
    
    n_agents, n_timesteps, _ = positions.shape
    
    # Handle mask shape - convert to (n_timesteps, n_agents)
    if player_mask.shape[0] == n_agents and player_mask.shape[1] == n_timesteps:
        player_mask = player_mask.T
    
    # Auto-calculate axis limits if not provided
    if xlim is None or ylim is None:
        # Gather all x, y coordinates from trajectories, goals, and initial positions
        all_x = np.concatenate([
            positions[:, :, 0].flatten(),
            goals[:, 0],
            init_positions[:, 0]
        ])
        all_y = np.concatenate([
            positions[:, :, 1].flatten(),
            goals[:, 1],
            init_positions[:, 1]
        ])
        
        # Calculate min and max with padding (15% of the range)
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_padding = max(0.15 * x_range, 0.5)  # At least 0.5 units of padding
        y_padding = max(0.15 * y_range, 0.5)
        
        if xlim is None:
            xlim = (x_min - x_padding, x_max + x_padding)
        if ylim is None:
            ylim = (y_min - y_padding, y_max + y_padding)
    
    # Color scheme
    ego_color = 'darkblue'
    masked_color = 'red'
    unmasked_color = 'gray'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_title('Agent Trajectories with Player Mask', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot ego agent's full trajectory in blue
    ego_traj = positions[ego_agent_id, :, :]
    if len(ego_traj) > 1:
        ax.plot(ego_traj[:, 0], ego_traj[:, 1], '-', 
               color=ego_color, alpha=0.9, linewidth=3, 
               label=f'Ego Agent {ego_agent_id}')
    
    # Plot other agents' trajectories with mask-based coloring
    for i in range(n_agents):
        if i == ego_agent_id:
            continue  # Skip ego agent (already plotted)
        
        agent_traj = positions[i, :, :]
        if len(agent_traj) <= 1:
            continue
        
        # Get mask for this agent across all timesteps
        agent_mask = player_mask[:, i]  # Shape: (n_timesteps,)
        
        # Segment trajectory based on mask changes
        # Convert boolean to int and find where it changes
        mask_int = agent_mask.astype(int)
        diffs = np.diff(mask_int)
        # Find indices where mask changes (add 1 because diff gives index of second element)
        change_indices = np.where(diffs != 0)[0] + 1
        # Create segment boundaries: [0, change1, change2, ..., n_timesteps]
        segment_boundaries = np.concatenate([[0], change_indices, [n_timesteps]])
        
        # Plot each segment
        for seg_idx in range(len(segment_boundaries) - 1):
            start_idx = segment_boundaries[seg_idx]
            end_idx = segment_boundaries[seg_idx + 1]
            
            if start_idx >= end_idx:
                continue
            
            # Determine color based on mask value at start of segment
            is_masked = agent_mask[start_idx]
            
            # Extract segment
            seg_traj = agent_traj[start_idx:end_idx, :]
            
            if len(seg_traj) > 1:
                color = masked_color if is_masked else unmasked_color
                alpha = 0.8 if is_masked else 0.4
                linewidth = 2.5 if is_masked else 1.5
                linestyle = '-' if is_masked else '--'
                
                ax.plot(seg_traj[:, 0], seg_traj[:, 1], 
                       linestyle, color=color, alpha=alpha, linewidth=linewidth,
                       label=f'Agent {i}' if seg_idx == 0 and is_masked else '')
    
    # Plot final positions
    for i in range(n_agents):
        final_pos = positions[i, -1, :]
        is_masked = player_mask[-1, i] if i < player_mask.shape[1] else False
        
        if i == ego_agent_id:
            # Ego agent: blue circle
            ax.scatter(final_pos[0], final_pos[1],
                      color=ego_color, s=100, marker='o', alpha=0.8,
                      edgecolors='black', linewidth=2)
        else:
            # Other agents: colored by mask status
            color = masked_color if is_masked else unmasked_color
            ax.scatter(final_pos[0], final_pos[1],
                      color=color, s=80, marker='o', alpha=0.8,
                      edgecolors='black', linewidth=1.5)
            # Add text label with mask indicator
            label_text = f'{i}*' if is_masked else f'{i}'
            ax.text(final_pos[0] + 0.1, final_pos[1] + 0.1, 
                   label_text, fontsize=10, ha='left', va='bottom', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Plot goals as stars
    for i in range(n_agents):
        goal_pos = goals[i]
        
        if i == ego_agent_id:  # Ego agent goal
            ax.scatter(goal_pos[0], goal_pos[1],
                      color=ego_color, s=200, marker='*', 
                      edgecolors='black', linewidth=2, alpha=0.8)
        else:  # Other agent goals
            # Check if agent is ever in mask to determine goal color
            ever_masked = np.any(player_mask[:, i]) if i < player_mask.shape[1] else False
            goal_color = masked_color if ever_masked else unmasked_color
            ax.scatter(goal_pos[0], goal_pos[1],
                      color=goal_color, s=150, marker='*', 
                      edgecolors='black', linewidth=1.5, alpha=0.6)
    
    plt.tight_layout()
    
    # Save as PNG
    print(f"Saving PNG to: {save_path}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ PNG created successfully!")

def plot_past_and_predicted_drone_agent_trajectories():
    pass