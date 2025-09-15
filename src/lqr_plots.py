import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional, List, Sequence
import jax.numpy as jnp
import json
import datetime


class LQRPlots:
    """Plotting utilities for LQR multi-agent trajectories, controls, and repulsion fields.

    This class mirrors the API style of `old_solvers/ilqr_plots.py` but is adapted to work
    directly with the current `Agent` class and arrays returned by `solve_single_agent`.
    """

    def __init__(
        self,
        agents: Sequence,
        goals: Sequence[jnp.ndarray],
        results: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
        dt: float,
        horizon: int,
        other_agent_positions_list: Optional[Sequence[Optional[jnp.ndarray]]] = None,
    ) -> None:
        """Initialize plotter.

        Args:
            agents: List of `Agent` instances.
            goals: List of goal positions for each agent, shape (2,) per agent.
            results: Per-agent tuples of (u_traj, x_traj).
            dt: Time step used in planning.
            horizon: Number of time steps.
            other_agent_positions_list: Optional list where each element is (T, M, 2)
                other agents' predicted positions used when solving agent i. If None,
                repulsion plotting will be skipped.
        """
        self.agents = list(agents)
        self.goals = list(goals)
        self.results = list(results)
        self.dt = dt
        self.tsteps = horizon
        self.other_agent_positions_list = (
            list(other_agent_positions_list) if other_agent_positions_list is not None else None
        )

    def plot_trajectories(
        self,
        figsize: Tuple[int, int] = (10, 8),
        show_arrows: bool = True,
        arrow_interval: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        for i, agent in enumerate(self.agents):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            u_traj, x_traj = self.results[i]
            if x_traj is None:
                print(f"Warning: Agent {i} has no trajectory data")
                continue

            x_coords = x_traj[:, 0]
            y_coords = x_traj[:, 1]
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7,
                    label=f'Agent {i} trajectory')

            start_x, start_y = float(agent.x0[0]), float(agent.x0[1])
            ax.scatter(start_x, start_y, color=color, s=100, marker='o',
                       edgecolors='black', linewidth=2, zorder=5,
                       label=f'Agent {i} start')

            goal_x, goal_y = float(self.goals[i][0]), float(self.goals[i][1])
            ax.scatter(goal_x, goal_y, color=color, s=120, marker='*',
                       edgecolors='black', linewidth=2, zorder=5,
                       label=f'Agent {i} goal')

            if show_arrows and len(x_coords) > arrow_interval:
                for t in range(0, len(x_coords) - 1, arrow_interval):
                    if t + 1 < len(x_coords):
                        dx = x_coords[t + 1] - x_coords[t]
                        dy = y_coords[t + 1] - y_coords[t]
                        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                            ax.arrow(x_coords[t], y_coords[t], dx * 0.3, dy * 0.3,
                                     head_width=0.1, head_length=0.1,
                                     fc=color, ec=color, alpha=0.6)

            ax.annotate(f'Start {i}', (start_x, start_y),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)
            ax.annotate(f'Goal {i}', (goal_x, goal_y),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)

        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title('LQR Multi-Agent Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        all_x = []
        all_y = []
        for _, x_traj in self.results:
            if x_traj is not None:
                all_x.extend(x_traj[:, 0])
                all_y.extend(x_traj[:, 1])

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

    def plot_control_inputs(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        time_steps = jnp.arange(self.tsteps) * self.dt

        for i, (u_traj, _x_traj) in enumerate(self.results):
            color = colors[i % len(colors)]
            if u_traj is not None:
                axes[0].plot(time_steps[: len(u_traj)], u_traj[:, 0], color=color,
                             linewidth=2, label=f'Agent {i}')
                axes[1].plot(time_steps[: len(u_traj)], u_traj[:, 1], color=color,
                             linewidth=2, label=f'Agent {i}')

        axes[0].set_ylabel('a_x', fontsize=12)
        axes[0].set_title('Control Inputs Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('a_y', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Control inputs plot saved to {save_path}")
        plt.show()

    def plot_repulsion_fields(
        self,
        arrow_interval: int = 4,
        scale: float = 1.0,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot repulsive vectors along each agent's actual trajectory.

        Requires `other_agent_positions_list` to have been provided.
        """
        if self.other_agent_positions_list is None:
            print("No other agents' predicted positions provided; skipping repulsion plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, agent in enumerate(self.agents):
            color = colors[i % len(colors)]
            u_traj, x_traj = self.results[i]
            other_positions = self.other_agent_positions_list[i]
            if x_traj is None or other_positions is None:
                continue

            pos_i = x_traj[: self.tsteps, :2]  # (T, 2)
            # Compute repulsion vectors: sum_j alpha * (p_i - p_j) / (||...||^2 + eps)^(3/2)
            if other_positions.shape[1] > 0:
                d = pos_i[:, None, :] - other_positions  # (T, M, 2)
                r2 = jnp.sum(d ** 2, axis=2) + agent.repulsion_epsilon  # (T, M)
                rep_j = agent.repulsion_gain * d / (r2[:, :, None] ** (3.0 / 2.0))  # (T, M, 2)
                repel = jnp.sum(rep_j, axis=1)  # (T, 2)
            else:
                repel = jnp.zeros_like(pos_i)

            ax.plot(pos_i[:, 0], pos_i[:, 1], color=color, linewidth=2, alpha=0.6,
                    label=f'Agent {i} trajectory')

            # Quiver every few steps
            idxs = list(range(0, len(pos_i), max(1, arrow_interval)))
            ax.quiver(
                pos_i[idxs, 0], pos_i[idxs, 1],
                repel[idxs, 0] * scale, repel[idxs, 1] * scale,
                angles='xy', scale_units='xy', scale=1.0, color=color, alpha=0.7,
            )

            # Start/goal markers
            start_x, start_y = float(agent.x0[0]), float(agent.x0[1])
            goal_x, goal_y = float(self.goals[i][0]), float(self.goals[i][1])
            ax.scatter(start_x, start_y, color=color, s=80, marker='o', edgecolors='black', linewidth=1.5)
            ax.scatter(goal_x, goal_y, color=color, s=120, marker='*', edgecolors='black', linewidth=1.5)

        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title('Repulsion Fields Along Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Repulsion field plot saved to {save_path}")
        plt.show()

    def export_trajectories_to_json(self, save_path: str) -> None:
        """Export agents' trajectories and controls to a JSON file."""
        data = {}

        timestamps = [i * self.dt for i in range(self.tsteps)]

        for i, agent in enumerate(self.agents):
            u_traj, x_traj = self.results[i]
            agent_key = f"agent_{i}"
            data[agent_key] = {
                "positions": {},
                "trajectories": {},
                "controls": {}
            }

            if x_traj is not None:
                for t, timestamp in enumerate(timestamps):
                    if t < len(x_traj):
                        position = [float(x) for x in x_traj[t].tolist()]
                        data[agent_key]["positions"][f"timestamp_{t}"] = position
                        data[agent_key]["trajectories"][f"timestamp_{t}"] = position

            if u_traj is not None:
                for t, timestamp in enumerate(timestamps):
                    if t < len(u_traj):
                        control = [float(x) for x in u_traj[t].tolist()]
                        data[agent_key]["controls"][f"timestamp_{t}"] = control

        data["metadata"] = {
            "num_agents": len(self.agents),
            "time_step": self.dt,
            "total_time_steps": self.tsteps,
            "total_duration": self.tsteps * self.dt,
            "export_timestamp": datetime.datetime.now().isoformat(),
            "agent_initial_positions": [agent.x0.tolist() for agent in self.agents],
            "agent_goals": [goal.tolist() for goal in self.goals],
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Trajectory data exported to {save_path}")


