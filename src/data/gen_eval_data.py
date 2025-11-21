import json
import os
import sys
import random
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from load_config import load_config
from utils.goal_init import random_init
from utils.plot import plot_point_agent_trajs, plot_drone_agent_trajs

if __name__ == "__main__":
    config = load_config()
    n_agents = config.game.N_agents
    agent_type = config.game.agent_type
    opt_config = getattr(config.optimization, agent_type)
    x_dim = opt_config.state_dim
    pos_dim = x_dim // 2

    num_samples = 150 
    output_dir = f"src/data/{agent_type}_agent_data/eval_data_upto_{n_agents}p"
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
        init_ps, init_goals = random_init(n_agents, (-boundary_size, boundary_size), dims=pos_dim)
        out_file = os.path.join(output_dir, f"eval_data_sample_{sample_id:03d}.json")
        with open(out_file, "w") as f:
            json.dump({
                "n_agents": n_agents,
                "boundary_size": float(boundary_size),
                "init_ps": [arr.tolist() for arr in init_ps],
                "init_goals": [arr.tolist() for arr in init_goals],
            }, f, indent=2)
        
        # plot start and goal positions using appropriate plotting function based on agent type
        plot_path = os.path.join(output_dir, f"eval_data_sample_{sample_id:03d}.png")
        
        # Create a minimal trajectory with 2 timesteps (start and goal) for visualization
        trajs = np.array([[init_ps[i], init_goals[i]] for i in range(n_agents)])  # (n_agents, 2, pos_dim)
        
        # Call the appropriate plotting function based on agent type
        if agent_type == "point":
            plot_point_agent_trajs(
                trajs=trajs,
                goals=np.array(init_goals),
                init_points=np.array(init_ps),
                title=f"Eval Data Sample {sample_id:03d} - {n_agents} agents",
                show_legend=(n_agents <= 10),
                save_path=plot_path
            )
        elif agent_type == "drone":
            plot_drone_agent_trajs(
                trajs=trajs,
                goals=np.array(init_goals),
                init_points=np.array(init_ps),
                title=f"Eval Data Sample {sample_id:03d} - {n_agents} agents",
                show_legend=(n_agents <= 10),
                save_path=plot_path
            )
        else:
            # Fallback for unknown agent types
            print(f"Warning: Unknown agent type '{agent_type}', skipping plot.")



    

