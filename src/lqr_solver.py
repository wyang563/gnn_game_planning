import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
import datetime
import os
import random
from utils.utils import random_init
from utils.lqr_plots import LQRPlotter, plot_simulation_results

class Agent(iLQR):
    def __init__(
        self, 
        id: int,
        dt: float, 
        x_dim: int,
        u_dim: int, 
        Q: jnp.ndarray, 
        R: jnp.ndarray,
        horizon: int,
        x0: jnp.ndarray,
        u_traj: jnp.ndarray,
        goal: jnp.ndarray,
        device = devices("cpu")[0],
        max_velocity: float = 2.0,
        max_acceleration: float = 2.0,
        goal_threshold: float = 0.1,
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)
        # dynamics 
        self.id = id
        self.horizon = horizon
        self.x0 = x0
        self.u_traj = u_traj
        self.goal = goal
        self.device = device
        self.goal_threshold = goal_threshold
        self.x_traj: Optional[jnp.ndarray] = None  
        self.A_traj: Optional[jnp.ndarray] = None 
        self.B_traj: Optional[jnp.ndarray] = None

        # values for logging
        self.past_x_traj: List[jnp.ndarray] = []
        self.past_u_traj: List[jnp.ndarray] = []
        self.past_loss: List[float] = []

        # Game-theoretic parameters
        self.max_velocity = max_velocity  # v̄ for velocity clamping
        self.max_acceleration = max_acceleration  # ū for control clamping

        # methods
        self.jit_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.jit_solve = jit(self.solve, device=self.device)
        self.jit_loss = jit(self.loss, device=device)
        self.jit_linearize_loss = jit(self.linearize_loss, device=device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])
    
    # define loss functions
    def runtime_loss(self, xt, ut, ref_xt, other_xts):
        nav_loss: jnp.ndarray = jnp.sum(jnp.square(xt[:2]-ref_xt[:2]))

        collision_loss = 0.0
        if other_xts.shape[0] > 0:  
            # TODO: optimize this with JNP later in one operation
            for other_xt in other_xts:
                collision_loss += 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt[:2])))
            # normalize collision loss 
            collision_loss /= other_xts.shape[0]
            
        ctrl_loss: jnp.ndarray = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.5])))
        return nav_loss + collision_loss + ctrl_loss

    def loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs):
        per_t = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0))
        return per_t(x_traj, u_traj, ref_x_traj, other_x_trajs).sum() * self.dt

    def linearize_loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs):
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj = vmap(dldx, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)  
        b_traj = vmap(dldu, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs) 
        return a_traj, b_traj

    # helpers
    def get_ref_traj(self):
        return jnp.linspace(self.x0[:2], self.goal, self.horizon+1)[1:]
    
    def check_convergence(self) -> bool:
        return jnp.linalg.norm(self.x0[:2] - self.goal) < self.goal_threshold

class Simulator:
    def __init__(
        self, 
        n_agents: int,
        Q: jnp.ndarray,
        R: jnp.ndarray, 
        horizon: int, 
        dt: float, 
        init_arena_range: Tuple[float, float], 
        device: str = "cpu",
        max_acceleration: float = 2.0,
        max_velocity: float = 2.0,
        goal_threshold: float = 0.1,
        total_iters: int = 1000,
        init_ps: List[jnp.ndarray] = None,
        goals: List[jnp.ndarray] = None,
    ) -> None: 

        self.n_agents = n_agents
        self.Q = Q
        self.R = R
        self.horizon = horizon
        self.dt = dt
        self.timestep = 0
        self.init_arena_range = init_arena_range
        self.device = devices(device)[0]
        self.goal_threshold = goal_threshold
        self.agent_trajectories = None
        self.total_iters = total_iters 
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        self.agents: List[Agent] = []
        self.setup_sim(init_ps, goals)

    def setup_sim(self, init_ps: List[jnp.ndarray] = None, goals: List[jnp.ndarray] = None) -> None:
        if init_ps is None or goals is None:
            init_ps, goals = random_init(self.n_agents, self.init_arena_range)

        for i in range(self.n_agents):
            px, py = float(init_ps[i][0]), float(init_ps[i][1]) 
            x0 = jnp.array([px, py, 0, 0])
            u_traj = jnp.zeros((self.horizon, 2))
            goal_i = goals[i]

            agent = Agent(
                id=i,
                dt=self.dt, 
                x_dim=4, 
                u_dim=2,
                Q=self.Q,
                R=self.R,
                horizon=self.horizon,
                x0=x0,
                u_traj=u_traj,
                goal=goal_i,
                device=self.device,
                max_velocity=self.max_velocity,
                max_acceleration=self.max_acceleration,
                goal_threshold=self.goal_threshold
            )

            self.agents.append(agent)
    
    def get_other_player_trajectories(self) -> List[jnp.ndarray]:
        self.agent_trajectories = [agent.get_ref_traj() for agent in self.agents]

    def sim_step(self) -> None:
        # Store current states before horizon-T optimization
        current_states = [agent.x0.copy() for agent in self.agents]
        
        # Initialize control trajectories for optimization
        for agent in self.agents:
            agent.u_traj = jnp.zeros((self.horizon, 2))
        
        # Run horizon-T Nash game optimization iterations
        for _ in range(self.horizon):
            # Calculate game theoretic action for each agent
            for agent in self.agents:
                agent.x_traj, agent.A_traj, agent.B_traj = agent.jit_linearize_dyn(
                    agent.x0, agent.u_traj   
                )
            
            v_trajs = []
            for agent in self.agents:
                other_x_trajs_list = [other_agent.x_traj for other_agent in self.agents if other_agent.id != agent.id]
                if len(other_x_trajs_list) == 0:
                    other_x_trajs = jnp.zeros((self.horizon, 0, 4))
                else:
                    other_x_trajs = jnp.stack(other_x_trajs_list, axis=1)
                
                a_traj, b_traj = agent.jit_linearize_loss(
                    agent.x_traj, agent.u_traj, agent.get_ref_traj(), other_x_trajs
                )

                v_traj, _ = agent.jit_solve(
                    agent.A_traj, agent.B_traj, a_traj, b_traj
                )
                v_trajs.append(v_traj)
        
            # Update control trajectories for next iteration
            for i, agent in enumerate(self.agents):
                agent.u_traj += self.dt * v_trajs[i]
        
        for agent in self.agents:
            u0 = agent.u_traj[0]  
            agent.x0 = agent.dyn_step(agent.x0, u0)[0]  
            agent.past_x_traj.append(current_states[agent.id])  
            agent.past_u_traj.append(u0)  

            # calculate runtime loss for time step
            other_x_trajs_list = [other_agent.x0 for other_agent in self.agents if other_agent.id != agent.id]
            if len(other_x_trajs_list) == 0:
                other_x_trajs = jnp.zeros((0, 4))
            else:
                other_x_trajs = jnp.stack(other_x_trajs_list, axis=0)
            loss = agent.runtime_loss(agent.x0, u0, agent.get_ref_traj()[0], other_x_trajs)
            agent.past_loss.append(float(loss))
        
        self.timestep += self.dt
    
    def run(self) -> None:
        for _ in range(self.total_iters):
            self.sim_step()
        
        # check convergence
        for agent in self.agents:
            print(f"Agent {agent.id}: final_distance_to_goal={jnp.linalg.norm(agent.x0[:2] - agent.goal):.3f}, converged={agent.check_convergence()}")
    
    def plot_results(self, gif_interval: int = 100) -> dict:
        """
        Create plots for the simulation results.
        
        Args:
            timesteps: Number of timesteps for GIF (defaults to total_iters)
            gif_interval: Animation interval in milliseconds
            
        Returns:
            Dictionary with paths to created plot files
        """
        return plot_simulation_results(self, self.total_iters, gif_interval)

if __name__ == "__main__":
    # Test the new receding horizon Nash game Simulator
    print("=== Testing Receding Horizon Nash Game Simulator ===")
    
    # Simulation parameters
    N = 10  # Number of agents
    horizon = 20  # Optimization horizon (T steps)
    dt = 0.1  # Time step
    init_arena_range = (-5.0, 5.0)  # Initial position range
    goal_threshold = 0.2  # Convergence threshold
    device = "cpu"
    max_acceleration = 2.0
    max_velocity = 2.0
    total_iters = 50  # Total simulation steps

    # Cost weights (position, velocity, control)
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))  # Higher position weights
    R = jnp.eye(2) * 0.1  # Control cost

    print(f"Parameters: N={N}, horizon={horizon}, dt={dt}, total_iters={total_iters}")
    print(f"Cost weights - Q: {Q.diagonal()}, R: {R.diagonal()}")
    print()

    # fixed goals
    init_ps = [jnp.array([-2.0, 0.0]), jnp.array([2.0, 0.0])]
    goals = [jnp.array([2.0, 0.0]), jnp.array([-2.0, 0.0])]
    N = 2

    # Create simulator    
    simulator = Simulator(
        n_agents=N,
        Q=Q,
        R=R,
        horizon=horizon,
        dt=dt,
        init_arena_range=init_arena_range,
        device=device,
        max_acceleration=max_acceleration,
        max_velocity=max_velocity,
        goal_threshold=goal_threshold,
        total_iters=total_iters,
        init_ps=init_ps,
        goals=goals
    )

    # Print initial conditions
    print("Initial conditions:")
    for i, agent in enumerate(simulator.agents):
        print(f"  Agent {i}: pos=({agent.x0[0]:.2f}, {agent.x0[1]:.2f}), goal=({agent.goal[0]:.2f}, {agent.goal[1]:.2f})")
    print()

    # Run simulation with progress updates
    print("Running simulation...")
    simulator.run()
    
    # Create plots
    print("\nCreating plots...")
    plot_files = simulator.plot_results()
    print(f"Plots saved to: {plot_files}") 