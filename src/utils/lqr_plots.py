import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
import os
from datetime import datetime


class LQRPlotter:
    """Plotting utilities for LQR multi-agent simulations."""
    
    def __init__(self, 
                 agents: List,
                 arena_bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5),
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 100):
        """
        Initialize the plotter.
        
        Args:
            agents: List of Agent objects from the simulator
            arena_bounds: (x_min, x_max, y_min, y_max) for plot bounds
            figsize: Figure size for plots
            dpi: DPI for plots
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.arena_bounds = arena_bounds
        self.figsize = figsize
        self.dpi = dpi
        
        # Colors for different agents
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.n_agents))
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"outputs/{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_trajectory_gif(self, 
                            timesteps: int,
                            interval: int = 100,
                            filename: Optional[str] = None) -> str:
        """
        Create an animated GIF showing agent trajectories over time.
        
        Args:
            timesteps: Number of timesteps to animate
            interval: Animation interval in milliseconds
            filename: Output filename (if None, auto-generates with timestamp)
            
        Returns:
            Path to the created GIF file
        """
        if filename is None:
            filename = "lqr_trajectory_animation.gif"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Initialize empty lines and points for each agent
        lines = []
        points = []
        goal_points = []
        
        for i, agent in enumerate(self.agents):
            # Trajectory line
            line, = ax.plot([], [], color=self.colors[i], linewidth=2, 
                          alpha=0.7, label=f'Agent {i} Trajectory')
            lines.append(line)
            
            # Current position point
            point, = ax.plot([], [], 'o', color=self.colors[i], 
                           markersize=8, markeredgecolor='black', 
                           markeredgewidth=1, label=f'Agent {i}')
            points.append(point)
            
            # Goal point
            goal_point, = ax.plot([], [], 's', color=self.colors[i], 
                                markersize=10, markeredgecolor='black', 
                                markeredgewidth=2, alpha=0.8, label=f'Agent {i} Goal')
            goal_points.append(goal_point)
        
        # Set up the plot
        ax.set_xlim(self.arena_bounds[0], self.arena_bounds[1])
        ax.set_ylim(self.arena_bounds[2], self.arena_bounds[3])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Multi-Agent LQR Trajectory Animation')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        def animate(frame):
            # Clear previous data
            for line in lines:
                line.set_data([], [])
            for point in points:
                point.set_data([], [])
            
            # Update data for each agent
            for i, agent in enumerate(self.agents):
                if len(agent.past_x_traj) > 0:
                    # Get trajectory data up to current frame
                    trajectory_data = np.array(agent.past_x_traj[:min(frame + 1, len(agent.past_x_traj))])
                    if len(trajectory_data) > 0:
                        x_traj = trajectory_data[:, 0]
                        y_traj = trajectory_data[:, 1]
                        lines[i].set_data(x_traj, y_traj)
                        
                        # Current position
                        current_pos = agent.past_x_traj[min(frame, len(agent.past_x_traj) - 1)]
                        points[i].set_data([current_pos[0]], [current_pos[1]])
                
                # Goal position (static)
                goal_pos = agent.goal
                goal_points[i].set_data([goal_pos[0]], [goal_pos[1]])
            
            # Update title with current timestep
            ax.set_title(f'Multi-Agent LQR Trajectory Animation - Timestep {frame}')
            
            return lines + points + goal_points
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=timesteps, 
                                     interval=interval, blit=True, repeat=True)
        
        # Save as GIF
        print(f"Creating trajectory GIF: {filepath}")
        anim.save(filepath, writer='pillow', fps=10)
        plt.close(fig)
        
        return filepath
    
    def plot_trajectories(self, filename: Optional[str] = None) -> str:
        """
        Create a static plot showing all agent trajectories.
        
        Args:
            filename: Output filename (if None, auto-generates with timestamp)
            
        Returns:
            Path to the created image file
        """
        if filename is None:
            filename = "lqr_trajectories.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot trajectories for each agent
        for i, agent in enumerate(self.agents):
            if len(agent.past_x_traj) > 0:
                trajectory_data = np.array(agent.past_x_traj)
                x_traj = trajectory_data[:, 0]
                y_traj = trajectory_data[:, 1]
                
                # Plot trajectory line
                ax.plot(x_traj, y_traj, color=self.colors[i], linewidth=2, 
                       alpha=0.7, label=f'Agent {i} Trajectory')
                
                # Mark start and end points
                ax.plot(x_traj[0], y_traj[0], 'o', color=self.colors[i], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1,
                       label=f'Agent {i} Start')
                ax.plot(x_traj[-1], y_traj[-1], 's', color=self.colors[i], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1,
                       label=f'Agent {i} End')
                
                # Plot goal
                goal_pos = agent.goal
                ax.plot(goal_pos[0], goal_pos[1], 'X', color=self.colors[i], 
                       markersize=12, markeredgecolor='black', markeredgewidth=2,
                       label=f'Agent {i} Goal')
        
        # Set up the plot
        ax.set_xlim(self.arena_bounds[0], self.arena_bounds[1])
        ax.set_ylim(self.arena_bounds[2], self.arena_bounds[3])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Multi-Agent LQR Trajectories')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        # Save the plot
        print(f"Saving trajectory plot: {filepath}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def plot_accelerations(self, filename: Optional[str] = None) -> str:
        """
        Create a plot showing acceleration of each agent over time.
        
        Args:
            filename: Output filename (if None, auto-generates with timestamp)
            
        Returns:
            Path to the created image file
        """
        if filename is None:
            filename = "lqr_accelerations.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate accelerations for each agent
        for i, agent in enumerate(self.agents):
            if len(agent.past_u_traj) > 0:
                # Control inputs are accelerations in this case
                control_data = np.array(agent.past_u_traj)
                time_steps = np.arange(len(control_data))
                
                # Plot acceleration components
                ax.plot(time_steps, control_data[:, 0], color=self.colors[i], 
                       linewidth=2, linestyle='-', alpha=0.8, 
                       label=f'Agent {i} - X Acceleration')
                ax.plot(time_steps, control_data[:, 1], color=self.colors[i], 
                       linewidth=2, linestyle='--', alpha=0.8, 
                       label=f'Agent {i} - Y Acceleration')
        
        # Set up the plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Acceleration')
        ax.set_title('Multi-Agent LQR Accelerations Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot
        print(f"Saving acceleration plot: {filepath}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def plot_losses(self, filename: Optional[str] = None) -> str:
        """
        Create a plot showing loss values for each agent over time.
        
        Args:
            filename: Output filename (if None, auto-generates with timestamp)
            
        Returns:
            Path to the created image file
        """
        if filename is None:
            filename = "lqr_losses.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot losses for each agent
        for i, agent in enumerate(self.agents):
            if len(agent.past_loss) > 0:
                loss_data = np.array(agent.past_loss)
                time_steps = np.arange(len(loss_data))
                
                # Plot loss curve
                ax.plot(time_steps, loss_data, color=self.colors[i], 
                       linewidth=2, alpha=0.8, label=f'Agent {i}')
        
        # Set up the plot
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Loss Value')
        ax.set_title('Multi-Agent LQR Loss Values Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Use log scale for y-axis if losses vary widely
        if len(self.agents) > 0 and len(self.agents[0].past_loss) > 0:
            all_losses = np.concatenate([agent.past_loss for agent in self.agents if len(agent.past_loss) > 0])
            if np.max(all_losses) / np.min(all_losses[all_losses > 0]) > 10:  # If range is large
                ax.set_yscale('log')
        
        # Save the plot
        print(f"Saving loss plot: {filepath}")
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
    
    def plot_all(self, timesteps: int = 50, gif_interval: int = 100) -> dict:
        """
        Create all four types of plots.
        
        Args:
            timesteps: Number of timesteps for the GIF animation
            gif_interval: Animation interval in milliseconds
            
        Returns:
            Dictionary with paths to all created files
        """
        results = {}
        
        # Create trajectory GIF
        results['gif'] = self.create_trajectory_gif(timesteps, gif_interval)
        
        # Create static trajectory plot
        results['trajectories'] = self.plot_trajectories()
        
        # Create acceleration plot
        results['accelerations'] = self.plot_accelerations()
        
        # Create loss plot
        results['losses'] = self.plot_losses()
        
        print(f"\nAll plots created successfully!")
        print(f"GIF: {results['gif']}")
        print(f"Trajectories: {results['trajectories']}")
        print(f"Accelerations: {results['accelerations']}")
        print(f"Losses: {results['losses']}")
        
        return results


def plot_simulation_results(simulator, timesteps: int = 50, gif_interval: int = 100) -> dict:
    """
    Convenience function to plot results from a Simulator object.
    
    Args:
        simulator: Simulator object with agents
        timesteps: Number of timesteps for the GIF animation
        gif_interval: Animation interval in milliseconds
        
    Returns:
        Dictionary with paths to all created files
    """
    plotter = LQRPlotter(simulator.agents)
    return plotter.plot_all(timesteps, gif_interval)


if __name__ == "__main__":
    # Example usage
    print("LQR Plotting Utilities")
    print("This module provides plotting functions for LQR multi-agent simulations.")
    print("Use plot_simulation_results(simulator) to create all plots from a simulator object.")
