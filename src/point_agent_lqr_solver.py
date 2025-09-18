import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, devices
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from utils.utils import random_init_collision
from utils.point_agent_lqr_plots import LQRPlotter

class Agent(iLQR):
    def __init__(
        self, 
        id: int,
        dt: float, 
        x_dim: int,
        u_dim: int, 
        Q: jnp.ndarray, 
        R: jnp.ndarray,
        W: jnp.ndarray,
        horizon: int,
        x0: jnp.ndarray,
        u_traj: jnp.ndarray,
        goal: jnp.ndarray,
        device = devices("cpu")[0],
        goal_threshold: float = 0.1,
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)
        # simulation parameters 
        self.id = id
        self.horizon = horizon
        self.w1, self.w2, self.w3, self.w4 = W[0], W[1], W[2], W[3] 
        self.x0 = x0
        self.u_traj = u_traj
        self.goal = goal
        self.ref_traj = self.get_ref_traj()
        self.device = device
        self.goal_threshold = goal_threshold

        # dynamics values
        self.x_traj: Optional[jnp.ndarray] = None  
        self.A_traj: Optional[jnp.ndarray] = None 
        self.B_traj: Optional[jnp.ndarray] = None
        self.a_traj: Optional[jnp.ndarray] = None
        self.b_traj: Optional[jnp.ndarray] = None
        self.v_traj: Optional[jnp.ndarray] = None

        # values for logging
        self.loss_history: List[float] = []

        # methods
        self.jit_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.jit_solve = jit(self.solve, device=self.device)
        self.jit_loss = jit(self.loss, device=self.device)
        self.jit_linearize_loss = jit(self.linearize_loss, device=self.device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])
    
    # define loss functions
    def runtime_loss(self, xt, ut, ref_xt, other_xts):
        collision_loss = 0.0
        for other_xt in other_xts:
            collision_loss += self.w1 * jnp.exp(-self.w2 * jnp.sum(jnp.square(xt[:2] - other_xt[:2])))
        # normalize collision loss 
        collision_loss /= other_xts.shape[0]
        ctrl_loss: jnp.ndarray = self.w3 * jnp.sum(jnp.square(ut))
        nav_loss: jnp.ndarray = self.w4 * jnp.sum(jnp.square(xt[:2]-ref_xt[:2]))
        return nav_loss + collision_loss + ctrl_loss

    def loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs):
        runtime_loss_array = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)
        return runtime_loss_array.sum() * self.dt

    def linearize_loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs):
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj = vmap(dldx, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)  
        b_traj = vmap(dldu, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs) 
        return a_traj, b_traj

    # helpers
    def debug_agent(self):
        print(f"Agent {self.id}:")
        print(f"  x0={self.x0}")
        print(f"  goal={self.goal}")
        print(f"  ref_traj={self.ref_traj}")
        print(f"  x_traj={self.x_traj}")
        print(f"  u_traj={self.u_traj}")

    def get_ref_traj(self):
        return jnp.linspace(self.x0[:2], self.goal, self.horizon+1)[1:]
    
    def calculate_x_traj(self):
        return self.jit_linearize_dyn(self.x0, self.u_traj)[0]

    def check_convergence(self) -> bool:
        final_x_pos = self.calculate_x_traj()[-1, :2]
        return jnp.linalg.norm(final_x_pos - self.goal) < self.goal_threshold

class Simulator:
    def __init__(
        self, 
        n_agents: int,
        Q: jnp.ndarray,
        R: jnp.ndarray, 
        W: jnp.ndarray,
        horizon: int, 
        dt: float, 
        init_arena_range: Tuple[float, float], 
        device: str = "cpu",
        goal_threshold: float = 0.1,
        optimization_iters: int = 1000,
        step_size: float = 0.002,
        init_ps: List[jnp.ndarray] = None,
        goals: List[jnp.ndarray] = None,
    ) -> None: 

        self.n_agents = n_agents
        self.Q = Q
        self.R = R
        self.W = W
        self.horizon = horizon
        self.dt = dt
        self.timestep = 0
        self.init_arena_range = init_arena_range
        self.device = devices(device)[0]
        self.goal_threshold = goal_threshold
        self.agent_trajectories = None
        self.optimization_iters = optimization_iters 
        self.step_size = step_size
        self.agents: List[Agent] = []
        self.setup_sim(init_ps, goals)

    def setup_sim(self, init_ps: List[jnp.ndarray] = None, goals: List[jnp.ndarray] = None) -> None:
        if init_ps is None or goals is None:
            init_ps, goals = random_init_collision(self.n_agents, self.init_arena_range)

        for i in range(self.n_agents):
            px, py = float(init_ps[i][0]), float(init_ps[i][1]) 
            x0 = jnp.array([px, py, -0.8, 0.0])
            u_traj = jnp.zeros((self.horizon, 2))
            goal_i = goals[i]

            agent = Agent(
                id=i,
                dt=self.dt, 
                x_dim=4, 
                u_dim=2,
                Q=self.Q,
                R=self.R,
                W=self.W,
                horizon=self.horizon,
                x0=x0,
                u_traj=u_traj,
                goal=goal_i,
                device=self.device,
                goal_threshold=self.goal_threshold
            )

            self.agents.append(agent)
    
    def get_other_player_trajectories(self, agentId: int) -> List[jnp.ndarray]:
        return [agent.x_traj for agent in self.agents if agent.id != agentId]

    def step(self) -> None:
        for agent in self.agents:
            agent.x_traj, agent.A_traj, agent.B_traj = agent.jit_linearize_dyn(agent.x0, agent.u_traj)

        for agent in self.agents:
            other_player_trajectories_list = self.get_other_player_trajectories(agent.id)
            # transpose from (n,500,4) to (500,n,4) for broadcasting
            stacked_other_player_trajectories = jnp.stack(other_player_trajectories_list)
            other_player_trajectories = jnp.transpose(stacked_other_player_trajectories, (1, 0, 2))
            agent.a_traj, agent.b_traj = agent.jit_linearize_loss(agent.x_traj, 
                                                                  agent.u_traj,
                                                                  agent.ref_traj, 
                                                                  other_player_trajectories)

            loss_val = agent.jit_loss(agent.x_traj, agent.u_traj, agent.ref_traj, other_player_trajectories)
            agent.loss_history.append(loss_val)

            agent.v_traj, _ = agent.jit_solve(agent.A_traj, agent.B_traj, agent.a_traj, agent.b_traj)

        # print logging for loss values
        if int(self.timestep / self.dt) % int(self.optimization_iters/10) == 0:
            loss_str = f"iter[{int(self.timestep / self.dt)}/{self.optimization_iters}] | "
            for agent in self.agents:
                loss_str += f"Agent {agent.id}: {agent.loss_history[-1]:.3f} | "
            # print losses
            print(loss_str)

        for agent in self.agents:
            agent.u_traj += self.step_size * agent.v_traj

        self.timestep += self.dt
    
    def run(self) -> None:
        for _ in range(self.optimization_iters):
            self.step()
        
        # check convergence
        for agent in self.agents:
            print(f"Agent {agent.id}, converged={agent.check_convergence()}")

if __name__ == "__main__":
    # Test the new receding horizon Nash game Simulator
    print("=== Testing Receding Horizon Nash Game Simulator ===")
    
    # Simulation parameters
    N = 10  # Number of agents
    horizon = 200  # Optimization horizon (T steps)
    dt = 0.05  # Time step
    init_arena_range = (-10.0, 10.0)  # Initial position range
    goal_threshold = 0.2  # Convergence threshold
    device = "cpu"
    optimization_iters = 500  # Total simulation steps
    step_size = 0.002

    # Cost weights (position, velocity, control)
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # Higher position weights
    R = jnp.diag(jnp.array([0.01, 0.01]))
    W = jnp.array([20.0, 5.0, 0.1, 1.0]) # w1 = collision, w2 = collision cost exp decay, w3 = control, w4 = navigation

    # Create simulator    
    simulator = Simulator(
        n_agents=N,
        Q=Q,
        R=R,
        W=W,
        horizon=horizon,
        dt=dt,
        init_arena_range=init_arena_range,
        device=device,
        step_size=step_size,
        goal_threshold=goal_threshold,
        optimization_iters=optimization_iters,
    )

    # Print initial conditions
    print("Initial conditions:")
    for i, agent in enumerate(simulator.agents):
        print(f"  Agent {i}: pos=({agent.x0[0]:.2f}, {agent.x0[1]:.2f}), goal=({agent.goal[0]:.2f}, {agent.goal[1]:.2f})")
    print()

    # Run simulation with progress updates
    print("Running simulation...")
    simulator.run()
    
    # Generate plots
    print("\nGenerating visualization plots...")
    # Derive arena bounds from init_arena_range to ensure everything is in frame
    x_min, x_max = init_arena_range
    # Use same range for y for a square arena assumption
    arena_bounds = (x_min, x_max, x_min, x_max)
    plotter = LQRPlotter(simulator.agents, arena_bounds=arena_bounds)
    
    # Generate all static plots and optionally create GIF
    # Set create_gif=True to generate trajectory animation
    plotter.plot_all(create_gif=True, gif_interval=50)
    