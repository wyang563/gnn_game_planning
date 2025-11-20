from solver.point_agent import PointAgent
from solver.drone_agent import DroneAgent
import jax.numpy as jnp
from utils.plot import plot_point_agent_trajs, plot_point_agent_gif, plot_past_and_predicted_point_agent_trajectories, plot_drone_agent_trajs, plot_drone_agent_gif, plot_past_and_predicted_drone_agent_trajectories

def agent_type_to_state(agent_type: str):
    match agent_type:
        case "point":
            return 4, 2
        case "drone":
            return 6, 3
        case _:
            raise ValueError("unknown agent type")

def agent_type_to_Q_R_matrices(agent_type: str):
    match agent_type:
        case "point":
            return jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001])), jnp.diag(jnp.array([0.01, 0.01]))
        case "drone":
            return jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.001, 0.001, 0.001])), jnp.diag(jnp.array([0.01, 0.01, 0.01]))
        case _:
            raise ValueError("unknown agent type")

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
            return plot_point_agent_trajs, plot_point_agent_gif, plot_past_and_predicted_point_agent_trajectories
        case "drone":
            return plot_drone_agent_trajs, plot_drone_agent_gif, plot_past_and_predicted_drone_agent_trajectories
        case _:
            raise ValueError("unknown agent type")