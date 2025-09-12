import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from lqr_plots import LQRPlotter
import datetime
import os
import random

class Agent(iLQR):
    def __init__(
        self,
        id: int,
        dt: float,
        x_dim: int,
        u_dim: int,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        x0: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_traj: jnp.ndarray,
        device: str = "cpu",
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)

        self.id: int = id
        self.x0: jnp.ndarray = x0
        self.u_traj: jnp.ndarray = u_traj
        self.ref_traj: jnp.ndarray = ref_traj  # shape (T, 2) for (x, y)
        self.x_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim)
        self.A_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim, x_dim)
        self.B_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim, u_dim)

        # Convert device string to JAX device object
        if isinstance(device, str):
            device = devices(device)[0]
        
        self.jit_linearize_dyn = jit(self.linearize_dyn, device=device)
        self.jit_solve_ilqr = jit(self.solve, device=device)
        self.jit_loss = jit(self.loss, device=device)
        self.jit_linearize_loss = jit(self.linearize_loss, device=device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([
            ut[0] * jnp.cos(xt[2]),
            ut[0] * jnp.sin(xt[2]),
            ut[1]
        ])
    
    def runtime_loss(
        self,
        xt: jnp.ndarray,
        ut: jnp.ndarray,
        ref_xt: jnp.ndarray,
        other_xts: jnp.ndarray,
    ) -> jnp.ndarray:
        nav_loss: jnp.ndarray = jnp.sum(jnp.square(xt[:2]-ref_xt))

        collision_loss = 0.0
        if other_xts.shape[0] > 0:  
            # TODO: optimize this with JNP later in one operation
            for other_xt in other_xts:
                collision_loss += 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt[:2])))
            # normalize collision loss 
            collision_loss /= other_xts.shape[0]
            
        ctrl_loss: jnp.ndarray = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.01])))
        return nav_loss + collision_loss + ctrl_loss

    def loss(
        self,
        x_traj: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_x_traj: jnp.ndarray,
        other_x_trajs: jnp.ndarray,
    ) -> jnp.ndarray:
        per_t = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0))
        return per_t(x_traj, u_traj, ref_x_traj, other_x_trajs).sum() * self.dt

    def linearize_loss(
        self,
        x_traj: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_x_traj: jnp.ndarray,
        other_x_trajs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj = vmap(dldx, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)  
        b_traj = vmap(dldu, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs) 
        return a_traj, b_traj

class LQRSolver(iLQR):
    def __init__(self, 
                 n_agents: int = 2,
                 horizon: int = 30,
                 init_ps: List[jnp.ndarray] = None, # initial positions (x, y, theta)
                 init_us: List[jnp.ndarray] = None, # initial controls (v, omega)
                 goals: List[jnp.ndarray] = None,
                 dt: float = 0.1,
                 tsteps: int = 100,
                 Q: jnp.ndarray = jnp.eye(3), # penalties for deviation from ideal path
                 R: jnp.ndarray = jnp.eye(2), # penalties for deviation from ideal controls
                 device: str = "cpu"
                 ):
        self.n_agents: int = n_agents
        self.init_ps: List[jnp.ndarray] = init_ps
        self.init_us: List[jnp.ndarray] = init_us
        self.goals: List[jnp.ndarray] = goals
        self.horizon: int = horizon
        self.dt: float = dt
        self.tsteps: int = tsteps
        self.agents: List[Agent] = []
        self.device: str = device
        
        # create agents
        for i in range(n_agents):
            goal = self.goals[i]
            x0 = self.init_ps[i]
            u_traj = jnp.tile(self.init_us[i], reps=(tsteps, 1))
            ref_traj = jnp.linspace(x0[:2], goal, tsteps+1)[1:]
            agent = Agent(i, dt, x_dim=3, u_dim=2, Q=Q, R=R, x0=x0, u_traj=u_traj, ref_traj=ref_traj, device=device)
            self.agents.append(agent)
    
    def solve_state(self, num_iters: int = 200, step_size: float = 0.002):
        for iter in range(num_iters + 1):
            for agent in self.agents:
                agent.x_traj, agent.A_traj, agent.B_traj = agent.jit_linearize_dyn(
                    agent.x0, agent.u_traj   
                )
            
            v_trajs = []  
            for agent in self.agents:
                other_x_trajs_list = [other_agent.x_traj for other_agent in self.agents if other_agent.id != agent.id]
                
                if len(other_x_trajs_list) == 0:
                    other_x_trajs = jnp.zeros((self.tsteps, 0, 3))
                else:
                    other_x_trajs = jnp.stack(other_x_trajs_list, axis=1)
                
                a_traj, b_traj = agent.jit_linearize_loss(
                    agent.x_traj, agent.u_traj, agent.ref_traj, other_x_trajs
                )
                v_traj, _ = agent.jit_solve_ilqr(
                    agent.A_traj, agent.B_traj, a_traj, b_traj
                )
                v_trajs.append(v_traj)
                
                if iter % max(1, int(num_iters/10)) == 0:
                    loss = agent.jit_loss(
                        agent.x_traj, agent.u_traj, agent.ref_traj, other_x_trajs
                    )
                    print(f"iter[{iter:3d}/{num_iters}] Agent {agent.id}: loss = {loss:.6f}")
            
            for i, agent in enumerate(self.agents):
                agent.u_traj += step_size * v_trajs[i]


def random_init(n_agents: int, 
                init_position_range: Tuple[float, float]) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    init_ps = []
    init_us = []
    goals = []
    
    min_pos, max_pos = init_position_range
    pos_range = max_pos - min_pos
    
    # Minimum distance between agents to avoid initial collisions
    min_distance = 0.5 * pos_range / n_agents  # Scale with number of agents
    
    max_tries = 1000
    
    for _ in range(n_agents):
        # Generate initial position
        init_pos = None
        for _ in range(max_tries):
            x = random.uniform(min_pos, max_pos)
            y = random.uniform(min_pos, max_pos)
            theta = random.uniform(-jnp.pi, jnp.pi)
            
            candidate_pos = jnp.array([x, y, theta])
            
            # Check minimum distance from other agents
            too_close = False
            for existing_pos in init_ps:
                distance = jnp.linalg.norm(candidate_pos[:2] - existing_pos[:2])
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                init_pos = candidate_pos
                break
        
        init_ps.append(init_pos)
        
        # Generate initial control (random velocity and angular velocity)
        v = random.uniform(0.3, 1.2)  # Linear velocity
        omega = random.uniform(-0.5, 0.5)  # Angular velocity
        init_us.append(jnp.array([v, omega]))
        
        # Generate goal position (different from initial position)
        goal = None
        for _ in range(max_tries):
            goal_x = random.uniform(min_pos, max_pos)
            goal_y = random.uniform(min_pos, max_pos)
            candidate_goal = jnp.array([goal_x, goal_y])
            
            # Ensure goal is far enough from initial position
            distance_to_start = jnp.linalg.norm(candidate_goal - init_pos[:2])
            if distance_to_start > min_distance:
                goal = candidate_goal
                break
        
        goals.append(goal)
    
    return init_ps, init_us, goals


if __name__ == "__main__":
    # parameters we can toggle
    n_agents = 3
    solver_iters = 200  # Increased for better convergence
    step_size = 0.002
    tsteps = 100
    dt = 0.05
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.01]))  # State penalties
    R = jnp.diag(jnp.array([1.0, 0.01]))       # Control penalties

    init_ps, init_us, goals = random_init(n_agents=n_agents, init_position_range=(-2.0, 2.0))
    
    # Create solver
    solver = LQRSolver(
        n_agents=n_agents,
        init_ps=init_ps,
        init_us=init_us,
        goals=goals,
        dt=dt,
        tsteps=tsteps,
        Q=Q,
        R=R,
        device="cpu"
    )
    
    # Solve the Nash equilibrium
    solver.solve(num_iters=solver_iters, step_size=step_size)
    
    print("\nSolution completed!")
    
    # Create plotter and generate visualizations
    plotter = LQRPlotter(solver)
    
    output_prefix = f"outputs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(output_prefix, exist_ok=True)

    print("Plotting trajectories...")
    plotter.plot_trajectories(save_path=f"{output_prefix}/lqr_trajectories.png")
    
    # print("Creating animation...")
    # plotter.plot_trajectory_animation(save_path=f"{output_prefix}/lqr_animation.gif")
    
    print("Plotting control inputs...")
    plotter.plot_control_inputs(save_path=f"{output_prefix}/lqr_controls.png")
    
    print("Plotting positions over time...")
    plotter.plot_agent_positions_over_time(save_path=f"{output_prefix}/lqr_positions.png")
    
    # Export trajectory data to JSON
    print("Exporting trajectory data to JSON...")
    plotter.export_trajectories_to_json(save_path=f"{output_prefix}/trajectories_detailed.json")
    
    # Print final positions
    print(f"\nFinal positions:")
    for i, agent in enumerate(solver.agents):
        if agent.x_traj is not None:
            final_pos = agent.x_traj[-1][:2]
            goal = goals[i]
            distance = jnp.linalg.norm(final_pos - goal)
            print(f"  Agent {i}: pos={final_pos}, goal={goal}, dist={distance:.3f}")




            
            
            

