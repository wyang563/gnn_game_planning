import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional, List
import jax.numpy as jnp


class LQRPlotter:
    """Plotting utilities for LQR multi-agent trajectory visualization."""
    
    def __init__(self, solver):
        """
        Initialize plotter with a reference to the LQRSolver.
        
        Args:
            solver: LQRSolver instance containing agents and their trajectories
        """
        self.solver = solver
    
    def plot_trajectories(self, 
                         figsize: Tuple[int, int] = (10, 8),
                         show_arrows: bool = True,
                         arrow_interval: int = 5,
                         save_path: Optional[str] = None) -> None:
        """
        Plot the trajectories of all agents with start/goal markers.
        
        Args:
            figsize: Figure size (width, height)
            show_arrows: Whether to show direction arrows along trajectories
            arrow_interval: Show arrow every N time steps
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Color palette for different agents
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Marker styles for different agents
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        for i, agent in enumerate(self.solver.agents):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Get trajectory data
            x_traj = agent.x_traj
            if x_traj is None:
                print(f"Warning: Agent {agent.id} has no trajectory data")
                continue
                
            # Extract x, y coordinates
            x_coords = x_traj[:, 0]
            y_coords = x_traj[:, 1]
            
            # Plot trajectory line
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7, 
                   label=f'Agent {agent.id} trajectory')
            
            # Plot start point
            start_x, start_y = agent.x0[0], agent.x0[1]
            ax.scatter(start_x, start_y, color=color, s=100, marker='o', 
                      edgecolors='black', linewidth=2, zorder=5,
                      label=f'Agent {agent.id} start')
            
            # Plot goal point
            goal_x, goal_y = self.solver.goals[i][0], self.solver.goals[i][1]
            ax.scatter(goal_x, goal_y, color=color, s=100, marker='*', 
                      edgecolors='black', linewidth=2, zorder=5,
                      label=f'Agent {agent.id} goal')
            
            # Add direction arrows along trajectory
            if show_arrows and len(x_coords) > arrow_interval:
                for t in range(0, len(x_coords)-1, arrow_interval):
                    if t + 1 < len(x_coords):
                        dx = x_coords[t+1] - x_coords[t]
                        dy = y_coords[t+1] - y_coords[t]
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only draw if movement is significant
                            ax.arrow(x_coords[t], y_coords[t], dx*0.3, dy*0.3,
                                   head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.6)
            
            # Add agent ID labels at start and goal
            ax.annotate(f'Start {agent.id}', (start_x, start_y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)
            ax.annotate(f'Goal {agent.id}', (goal_x, goal_y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)
        
        # Set plot properties
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title('Multi-Agent Trajectory Planning', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        # Add some padding around the trajectories
        all_x = []
        all_y = []
        for agent in self.solver.agents:
            if agent.x_traj is not None:
                all_x.extend(agent.x_traj[:, 0])
                all_y.extend(agent.x_traj[:, 1])
        
        if all_x and all_y:
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def plot_trajectory_animation(self, 
                                 interval: int = 50,
                                 save_path: Optional[str] = None) -> None:
        """
        Create an animated plot showing agents moving along their trajectories.
        
        Args:
            interval: Animation frame interval in milliseconds
            save_path: Optional path to save the animation as GIF
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Initialize plot elements
        lines = []
        points = []
        goal_points = []
        
        for i, agent in enumerate(self.solver.agents):
            color = colors[i % len(colors)]
            
            # Plot full trajectory as background
            if agent.x_traj is not None:
                x_coords = agent.x_traj[:, 0]
                y_coords = agent.x_traj[:, 1]
                line, = ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1)
                lines.append(line)
                
                # Current position point
                point, = ax.plot([], [], color=color, marker='o', markersize=8, 
                               markeredgecolor='black', markeredgewidth=1)
                points.append(point)
                
                # Goal point
                goal_x, goal_y = self.solver.goals[i][0], self.solver.goals[i][1]
                goal_point, = ax.plot(goal_x, goal_y, color=color, marker='*', 
                                    markersize=15, markeredgecolor='black', markeredgewidth=1)
                goal_points.append(goal_point)
        
        def animate(frame):
            for i, agent in enumerate(self.solver.agents):
                if agent.x_traj is not None and frame < len(agent.x_traj):
                    x = agent.x_traj[frame, 0]
                    y = agent.x_traj[frame, 1]
                    points[i].set_data([x], [y])
            return points + goal_points
        
        # Get trajectory length
        max_length = max(len(agent.x_traj) for agent in self.solver.agents if agent.x_traj is not None)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Multi-Agent Trajectory Animation')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set axis limits
        all_x = []
        all_y = []
        for agent in self.solver.agents:
            if agent.x_traj is not None:
                all_x.extend(agent.x_traj[:, 0])
                all_y.extend(agent.x_traj[:, 1])
        
        if all_x and all_y:
            x_margin = (max(all_x) - min(all_x)) * 0.1
            y_margin = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=max_length, interval=interval, 
                           blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        return anim

    def plot_loss_curves(self, 
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> None:
        """
        Plot the loss curves for all agents over iterations.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, agent in enumerate(self.solver.agents):
            color = colors[i % len(colors)]
            # Note: This would need loss history to be stored during optimization
            # For now, this is a placeholder for future implementation
            ax.plot([], [], color=color, linewidth=2, label=f'Agent {agent.id}')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Agent Loss Curves', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss curves plot saved to {save_path}")
        
        plt.show()

    def plot_control_inputs(self, 
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> None:
        """
        Plot the control inputs (velocity and angular velocity) for all agents.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        time_steps = jnp.arange(self.solver.tsteps) * self.solver.dt
        
        for i, agent in enumerate(self.solver.agents):
            color = colors[i % len(colors)]
            
            if agent.u_traj is not None:
                # Plot linear velocity
                axes[0].plot(time_steps, agent.u_traj[:, 0], color=color, 
                           linewidth=2, label=f'Agent {agent.id}')
                
                # Plot angular velocity
                axes[1].plot(time_steps, agent.u_traj[:, 1], color=color, 
                           linewidth=2, label=f'Agent {agent.id}')
        
        axes[0].set_ylabel('Linear Velocity (v)', fontsize=12)
        axes[0].set_title('Control Inputs Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Angular Velocity (ω)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Control inputs plot saved to {save_path}")
        
        plt.show()

    def plot_agent_positions_over_time(self, 
                                     figsize: Tuple[int, int] = (12, 8),
                                     save_path: Optional[str] = None) -> None:
        """
        Plot the x and y positions of all agents over time.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        time_steps = jnp.arange(self.solver.tsteps) * self.solver.dt
        
        for i, agent in enumerate(self.solver.agents):
            color = colors[i % len(colors)]
            
            if agent.x_traj is not None:
                # Plot x position
                axes[0].plot(time_steps, agent.x_traj[:, 0], color=color, 
                           linewidth=2, label=f'Agent {agent.id}')
                
                # Plot y position
                axes[1].plot(time_steps, agent.x_traj[:, 1], color=color, 
                           linewidth=2, label=f'Agent {agent.id}')
        
        axes[0].set_ylabel('X Position', fontsize=12)
        axes[0].set_title('Agent Positions Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Y Position', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Positions plot saved to {save_path}")
        
        plt.show()
