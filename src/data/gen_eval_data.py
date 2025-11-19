import json
import os
import sys
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from load_config import load_config
from utils.goal_init import random_init

if __name__ == "__main__":
    config = load_config()
    n_agents = config.game.N_agents

    num_samples = 150 
    output_dir = f"src/data/eval_data_upto_{n_agents}p"
    os.makedirs(output_dir, exist_ok=True)

    selection_pool = []
    upper_bound_agents = n_agents
    for i in range(2, upper_bound_agents + 1):
        if i < 4:
            selection_pool.extend([i] * 1)
        elif 4 <= i < 6:
            selection_pool.extend([i] * 2)
        else:
            selection_pool.extend([i] * 3)

    for sample_id in tqdm(range(num_samples), total=num_samples, desc="Generating evaluation data"):
        n_agents = random.choice(selection_pool)
        boundary_size = n_agents**(0.7)  * 1.75
        init_ps, init_goals = random_init(n_agents, (-boundary_size, boundary_size))
        out_file = os.path.join(output_dir, f"eval_data_sample_{sample_id:03d}.json")
        with open(out_file, "w") as f:
            json.dump({
                "n_agents": n_agents,
                "boundary_size": float(boundary_size),
                "init_ps": [arr.tolist() for arr in init_ps],
                "init_goals": [arr.tolist() for arr in init_goals],
            }, f, indent=2)
        
        # plot start and goal positions
        plot_path = os.path.join(output_dir, f"eval_data_sample_{sample_id:03d}.png")
        fig, ax = plt.subplots(figsize=(6, 6))
        init_ps_arr = list(init_ps)
        init_goals_arr = list(init_goals)
        for agent_idx in range(n_agents):
            color = plt.cm.tab20(agent_idx % 20)
            # Plot start point
            ax.scatter(init_ps_arr[agent_idx][0], init_ps_arr[agent_idx][1], marker='o', color=color, label=f"Agent {agent_idx+1} Start" if agent_idx == 0 else "")
            # Plot goal point
            ax.scatter(init_goals_arr[agent_idx][0], init_goals_arr[agent_idx][1], marker='X', color=color, label=f"Agent {agent_idx+1} Goal" if agent_idx == 0 else "")
            # Optionally, connect start to goal
            ax.plot([init_ps_arr[agent_idx][0], init_goals_arr[agent_idx][0]],
                    [init_ps_arr[agent_idx][1], init_goals_arr[agent_idx][1]],
                    color=color, linestyle='--', linewidth=1, alpha=0.6)
        ax.set_title(f"Eval Data Sample {sample_id:03d} - {n_agents} agents")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        if n_agents <= 10:
            ax.legend(loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)



    

