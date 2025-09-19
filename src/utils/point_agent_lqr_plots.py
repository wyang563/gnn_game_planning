import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, Any
import os
import json
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
        self._update_agent_trajectories()
    
    def _update_agent_trajectories(self):
        """Update x_traj for all agents by calling calculate_x_traj."""
        for agent in self.agents:
            agent.x_traj = agent.calculate_x_traj()
    
    def _get_dynamic_bounds(self, padding: float = 0.1) -> Tuple[float, float, float, float]:
        """Compute axis bounds that include all starts, goals, and trajectories.
        Unifies these with provided arena_bounds and adds padding.
        """
        xs = []
        ys = []
        for agent in self.agents:
            # start
            xs.append(float(agent.x0[0]))
            ys.append(float(agent.x0[1]))
            # goal
            xs.append(float(agent.goal[0]))
            ys.append(float(agent.goal[1]))
            # trajectory, if available
            if hasattr(agent, 'x_traj') and agent.x_traj is not None:
                xs.extend(np.asarray(agent.x_traj[:, 0]).tolist())
                ys.extend(np.asarray(agent.x_traj[:, 1]).tolist())
        if not xs or not ys:
            # fallback to provided bounds
            return self.arena_bounds
        x_min_data, x_max_data = min(xs), max(xs)
        y_min_data, y_max_data = min(ys), max(ys)
        # unify with provided arena bounds
        x_min = min(x_min_data, self.arena_bounds[0])
        x_max = max(x_max_data, self.arena_bounds[1])
        y_min = min(y_min_data, self.arena_bounds[2])
        y_max = max(y_max_data, self.arena_bounds[3])
        # add padding
        x_pad = (x_max - x_min) * padding if x_max > x_min else 1.0 * padding
        y_pad = (y_max - y_min) * padding if y_max > y_min else 1.0 * padding
        return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    
    def plot_trajectories(self, save_path: Optional[str] = None):
        """
        Plot the trajectories of each agent from start to end with goal positions.
        
        Args:
            save_path: Optional custom save path, defaults to output_dir
        """
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, agent in enumerate(self.agents):
            color = self.colors[i]
            
            # Extract x,y positions from trajectory
            x_positions = agent.x_traj[:, 0]
            y_positions = agent.x_traj[:, 1]
            
            # Plot trajectory
            ax.plot(x_positions, y_positions, color=color, linewidth=2, 
                   label=f'Agent {agent.id} trajectory', alpha=0.8)
            
            # Plot start position
            ax.scatter(agent.x0[0], agent.x0[1], color=color, s=100, 
                      marker='o', edgecolor='black', linewidth=2, 
                      label=f'Agent {agent.id} start')
            
            # Plot goal position
            ax.scatter(agent.goal[0], agent.goal[1], color=color, s=150, 
                      marker='*', edgecolor='black', linewidth=2,
                      label=f'Agent {agent.id} goal')
        
        bounds = self._get_dynamic_bounds(padding=0.08)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Trajectories')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'lqr_trajectories.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Trajectories plot saved to: {save_path}")
    
    def plot_accelerations(self, save_path: Optional[str] = None):
        """
        Plot the magnitude of each agent's acceleration over time.
        
        Args:
            save_path: Optional custom save path, defaults to output_dir
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, agent in enumerate(self.agents):
            color = self.colors[i]
            
            # Calculate acceleration magnitudes
            u_x = agent.u_traj[:, 0]
            u_y = agent.u_traj[:, 1]
            acceleration_magnitudes = jnp.sqrt(u_x**2 + u_y**2)
            
            # Simulation step vector (0 to horizon)
            simulation_steps = jnp.arange(len(acceleration_magnitudes))
            
            ax.plot(simulation_steps, acceleration_magnitudes, color=color, 
                   linewidth=2, label=f'Agent {agent.id}')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Acceleration Magnitude')
        ax.set_title('Agent Accelerations Over Simulation Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'lqr_accelerations.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Accelerations plot saved to: {save_path}")
    
    def plot_losses(self, save_path: Optional[str] = None):
        """
        Plot each agent's loss values over time.
        
        Args:
            save_path: Optional custom save path, defaults to output_dir
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, agent in enumerate(self.agents):
            color = self.colors[i]
            
            # Simulation step vector based on loss history length
            simulation_steps = jnp.arange(len(agent.loss_history))
            
            ax.plot(simulation_steps, agent.loss_history, color=color, 
                   linewidth=2, label=f'Agent {agent.id}')
        
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Loss Value')
        ax.set_title('Agent Loss Values Over Simulation Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Removed log scale for linear visualization
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'lqr_losses.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Losses plot saved to: {save_path}")
    
    def create_trajectory_gif(self, save_path: Optional[str] = None, 
                            interval: int = 100):
        """
        Create an animated GIF showing agents moving toward their goals over time.
        
        Args:
            save_path: Optional custom save path, defaults to output_dir
            interval: Animation interval in milliseconds
        """
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get maximum trajectory length
        max_traj_len = max(len(agent.x_traj) for agent in self.agents)
        
        # Initialize plot elements
        lines = []
        points = []
        goals = []
        
        for i, agent in enumerate(self.agents):
            color = self.colors[i]
            
            # Trajectory line (will show entire current trajectory each frame)
            line, = ax.plot([], [], color=color, linewidth=2, alpha=0.7,
                           label=f'Agent {agent.id}')
            lines.append(line)
            
            # Current position point
            point, = ax.plot([], [], color=color, marker='o', markersize=8,
                            markeredgecolor='black', markeredgewidth=1)
            points.append(point)
            
            # Goal position (static)
            goal = ax.scatter(agent.goal[0], agent.goal[1], color=color, s=150,
                             marker='*', edgecolor='black', linewidth=2,
                             alpha=0.8)
            goals.append(goal)
        
        bounds = self._get_dynamic_bounds(padding=0.08)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Trajectories Animation')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        def animate(frame):
            for i, agent in enumerate(self.agents):
                if frame < len(agent.x_traj):
                    # Update current position
                    x_pos = agent.x_traj[frame, 0]
                    y_pos = agent.x_traj[frame, 1]
                    points[i].set_data([x_pos], [y_pos])
                    
                    # Show entire trajectory up to current frame
                    x_trajectory = agent.x_traj[:frame+1, 0]
                    y_trajectory = agent.x_traj[:frame+1, 1]
                    lines[i].set_data(x_trajectory, y_trajectory)
                else:
                    # Keep final position if trajectory is shorter
                    final_x = agent.x_traj[-1, 0]
                    final_y = agent.x_traj[-1, 1]
                    points[i].set_data([final_x], [final_y])
                    lines[i].set_data(agent.x_traj[:, 0], agent.x_traj[:, 1])
            
            return lines + points
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=max_traj_len,
                                     interval=interval, blit=True, repeat=True)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'lqr_trajectories.gif')
        
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=1000//interval)
        plt.close()
        print(f"Trajectory animation saved to: {save_path}")
    
    def plot_all(self, create_gif: bool = False, gif_interval: int = 100):
        """
        Generate all plots and optionally the trajectory GIF.
        
        Args:
            create_gif: Whether to create the trajectory animation GIF
            gif_interval: Animation interval for GIF in milliseconds
        """
        print(f"Generating plots in directory: {self.output_dir}")
        
        # Generate static plots
        self.plot_trajectories()
        self.plot_accelerations()
        self.plot_losses()
        
        # Optionally generate GIF
        if create_gif:
            print("Creating trajectory animation...")
            self.create_trajectory_gif(interval=gif_interval)
        
        print("All plots generated successfully!")
    
    def dump_simulation_data(self, simulator, save_path: Optional[str] = None):
        """
        Dump simulation data to JSON file with metadata and agent trajectories.
        
        Args:
            simulator: Simulator object containing agents and simulation parameters
            save_path: Optional custom save path, defaults to output_dir
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'simulation_data.json')
        
        # Prepare metadata
        metadata = {
            "simulation_info": {
                "timestamp": datetime.now().isoformat(),
                "n_agents": int(simulator.n_agents),
                "horizon": int(simulator.horizon),
                "dt": float(simulator.dt),
                "optimization_iters": int(simulator.optimization_iters),
                "step_size": float(simulator.step_size),
                "goal_threshold": float(simulator.goal_threshold),
                "init_arena_range": [float(x) for x in simulator.init_arena_range],
                "device": str(simulator.device)
            },
            "cost_parameters": {
                "Q": [float(x) for x in np.diag(simulator.Q)],
                "R": [float(x) for x in np.diag(simulator.R)],
                "W": [float(x) for x in np.asarray(simulator.W)]
            }
        }
        
        # Prepare agent data
        agents_data = {}
        
        for agent in self.agents:
            # Convert JAX arrays to regular numpy arrays and then to lists
            x_traj = np.asarray(agent.x_traj).tolist()
            u_traj = np.asarray(agent.u_traj).tolist()
            
            # Convert loss history to regular Python floats
            loss_history = [float(loss) for loss in agent.loss_history]
            
            # Prepare timestep data
            timestep_data = {}
            for t in range(len(x_traj)):
                timestep_data[str(t)] = {
                    "x_traj": x_traj[t],
                    "u_traj": u_traj[t],
                    "loss": loss_history[t] if t < len(loss_history) else None
                }
            
            agents_data[f"agent_{agent.id}"] = {
                "agent_info": {
                    "id": agent.id,
                    "x0": [float(x) for x in np.asarray(agent.x0)],
                    "goal": [float(x) for x in np.asarray(agent.goal)],
                    "converged": bool(agent.check_convergence())
                },
                "timesteps": timestep_data
            }
        
        # Combine metadata and agent data
        simulation_data = {
            "metadata": metadata,
            "agents": agents_data
        }
        
        # Save to JSON file
        with open(save_path, 'w') as f:
            json.dump(simulation_data, f, indent=2)
        
        print(f"Simulation data saved to: {save_path}")
    
    def plot_all(self, create_gif: bool = False, gif_interval: int = 100, dump_data: bool = True, simulator=None):
        """
        Generate all plots and optionally the trajectory GIF and simulation data dump.
        
        Args:
            create_gif: Whether to create the trajectory animation GIF
            gif_interval: Animation interval for GIF in milliseconds
            dump_data: Whether to dump simulation data to JSON
            simulator: Simulator object (required if dump_data=True)
        """
        print(f"Generating plots in directory: {self.output_dir}")
        
        # Generate static plots
        self.plot_trajectories()
        self.plot_accelerations()
        self.plot_losses()
        
        # Optionally generate GIF
        if create_gif:
            print("Creating trajectory animation...")
            self.create_trajectory_gif(interval=gif_interval)
        
        # Optionally dump simulation data
        if dump_data and simulator is not None:
            print("Dumping simulation data to JSON...")
            self.dump_simulation_data(simulator)
        elif dump_data and simulator is None:
            print("Warning: dump_data=True but no simulator provided. Skipping data dump.")
        
        print("All plots generated successfully!")
    