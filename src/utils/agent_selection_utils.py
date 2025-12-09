from solver.point_agent import PointAgent
from solver.drone_agent import DroneAgent
import jax.numpy as jnp
from utils.plot import plot_point_agent_trajs, plot_point_agent_gif, plot_past_and_predicted_point_agent_trajectories, plot_drone_agent_trajs, plot_drone_agent_gif, plot_past_and_predicted_drone_agent_trajectories, plot_point_agent_trajs_gif, plot_drone_agent_trajs_gif, plot_point_agent_mask_png, plot_drone_agent_mask_png, plot_point_agent_mask_timesteps_png

def agent_type_to_agent_class(agent_type: str):
    match agent_type:
        case "point":
            return PointAgent
        case "drone":
            return DroneAgent
        case _:
            raise ValueError("unknown agent type")

def agent_type_to_plot_functions(agent_type: str):
    match agent_type:
        case "point":
            return {"plot_traj": plot_point_agent_trajs, "plot_traj_gif": plot_point_agent_trajs_gif, "plot_mask_gif": plot_point_agent_gif, "plot_mask_png": plot_point_agent_mask_png, "plot_past_and_predicted_traj": plot_past_and_predicted_point_agent_trajectories, "plot_mask_timesteps_png": plot_point_agent_mask_timesteps_png}
        case "drone":
            return {"plot_traj": plot_drone_agent_trajs, "plot_traj_gif": plot_drone_agent_trajs_gif, "plot_mask_gif": plot_drone_agent_gif, "plot_mask_png": plot_drone_agent_mask_png, "plot_past_and_predicted_traj": plot_past_and_predicted_drone_agent_trajectories}
        case _:
            raise ValueError("unknown agent type")