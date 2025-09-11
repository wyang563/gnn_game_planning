import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lqrax_solver import LQRaxNashSolver


def run_and_export(n_agents: int,
                   horizon: int,
                   dt: float,
                   goals_xy,
                   init_states,
                   json_out: Path,
                   plot_out: Path):
    """
    Run the QP solver, plot trajectories, and export state/control JSON.

    Args:
        n_agents: number of agents
        horizon: planning horizon
        dt: timestep
        goals_xy: list[[gx, gy], ...] length n_agents
        init_states: list[[px0, py0, vx0, vy0], ...] length n_agents
        json_out: path to write JSON dump
        plot_out: path to save trajectory plot
    """
    device = torch.device("cpu")

    solver = LQRaxNashSolver(n_agents=n_agents, horizon=horizon, dt=dt)

    initial_states = torch.tensor(init_states, dtype=torch.float32, device=device)
    goals = torch.tensor(goals_xy, dtype=torch.float32, device=device)

    print("Solving Nash equilibrium...")
    trajectories = solver.solve_nash_equilibrium(initial_states, goals)

    if trajectories is None:
        print("Failed to find solution")
        return

    # Extract controls from trajectories
    controls = solver.extract_controls(trajectories)

    # Prepare JSON structure
    data = {
        "n_agents": n_agents,
        "horizon": horizon,
        "dt": dt,
        "goals": goals.cpu().tolist(),
        "initial_states": initial_states.cpu().tolist(),
        "trajectories": {
            str(i): {
                "positions": trajectories[i, :, 0:2].cpu().tolist(),
                "velocities": trajectories[i, :, 2:4].cpu().tolist(),
                "controls": controls[i].cpu().tolist(),
            }
            for i in range(n_agents)
        },
    }

    # Write JSON
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote JSON to {json_out}")

    # Plot positions over time for each agent
    plt.figure(figsize=(7, 6))
    for i in range(n_agents):
        pos = trajectories[i, :, 0:2].cpu().numpy()
        plt.plot(pos[:, 0], pos[:, 1], marker="o", label=f"Agent {i}")
        # mark start and goal
        plt.scatter(pos[0, 0], pos[0, 1], c="k", marker="x")
        plt.scatter(goals[i, 0].item(), goals[i, 1].item(), c="k", marker="*")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Agent trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, ls=":", alpha=0.6)

    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out, bbox_inches="tight")
    print(f"Saved plot to {plot_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Run QP solver and plot/export trajectories")
    p.add_argument("--n_agents", type=int, default=2)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--goals", type=str, default="[[5,3],[2,5]]",
                   help="JSON list of goal positions per agent, e.g. [[5,3],[2,5]]")
    p.add_argument("--init_states", type=str, default="[[0,0,0,0],[2,2,0,0]]",
                   help="JSON list of initial states [px,py,vx,vy] per agent")
    p.add_argument("--json_out", type=str, default="outputs/trajectories.json")
    p.add_argument("--plot_out", type=str, default="outputs/trajectories.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    goals = json.loads(args.goals)
    init_states = json.loads(args.init_states)
    run_and_export(
        n_agents=args.n_agents,
        horizon=args.horizon,
        dt=args.dt,
        goals_xy=goals,
        init_states=init_states,
        json_out=Path(args.json_out),
        plot_out=Path(args.plot_out),
    )


