import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from old_solvers.ilqr_plots import LQRPlotter
import datetime
import os
import random
from utils.utils import random_init

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

        # past trajectories/controls
        self.past_x_traj: List[jnp.ndarray] = []
        self.past_u_traj: List[jnp.ndarray] = []

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
    def __init__(self, 
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
                 total_iters: int = 1000) -> None: 
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
        self.setup_sim()

    def setup_sim(self) -> None:
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
                max_acceleration=self.max_acceleration
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
        for iter in range(self.horizon):
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
        
        # Apply only the FIRST control action and update real-world state
        for agent in self.agents:
            # Apply first control action from optimized trajectory
            u0 = agent.u_traj[0]  # First control action
            
            # Update state using dynamics (single step)
            agent.x0 = agent.dyn_step(agent.x0, u0)[0]  # Get new state from RK4 integration
            
            # Store past trajectories for analysis
            agent.past_x_traj.append(current_states[agent.id])  # Store previous state
            agent.past_u_traj.append(u0)  # Store applied control
        
        self.timestep += self.dt
    
    def run(self) -> None:
        for _ in range(self.total_iters):
            self.sim_step()
        
        # check convergence
        for agent in self.agents:
            print(f"Agent {agent.id}: final_distance_to_goal={jnp.linalg.norm(agent.x0[:2] - agent.goal):.3f}, converged={agent.check_convergence()}")

if __name__ == "__main__":
    # Simple multi-agent demo: create N agents, compute trajectories, and check convergence
    N = 3
    horizon = 40
    dt = 0.1
    init_arena_range = (-5.0, 5.0)
    goal_threshold = 0.1
    device = "cpu"
    max_acceleration = 2.0
    max_velocity = 2.0
    goal_threshold = 0.1

    # Cost weights
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))
    R = jnp.eye(2) * 0.01

    # Create simulator    
    simulator = Simulator(
        n_agents=N,
        Q=Q,
        R=R,
        horizon=horizon,
        dt=dt,
        init_arena_range=init_arena_range,
        device=device,
        goal_threshold=goal_threshold
    )

    simulator.run()