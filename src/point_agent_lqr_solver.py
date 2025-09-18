import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from utils.utils import random_init_collision
from utils.point_agent_lqr_plots import LQRPlotter
from tqdm import tqdm

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
        device = None,
        goal_threshold: float = 0.1,
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)
        # simulation parameters 
        self.id = id
        self.horizon = horizon
        # Prefer CUDA if available; fallback to CPU
        if device is None:
            cuda_devs = jax.devices("cuda")
            self.device = cuda_devs[0] if len(cuda_devs) > 0 else jax.devices("cpu")[0]
        else:
            self.device = device

        # Place parameters and state on target device once to avoid repeated transfers
        W_dev = jax.device_put(W, self.device)
        self.w1, self.w2, self.w3, self.w4 = W_dev[0], W_dev[1], W_dev[2], W_dev[3]
        self.x0 = jax.device_put(x0, self.device)
        self.u_traj = jax.device_put(u_traj, self.device)
        self.goal = jax.device_put(goal, self.device)
        self.ref_traj = self.get_ref_traj()
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
        # calculate collision loss
        current_position = xt[:2]
        squared_distances = jnp.sum(jnp.square(current_position - other_xts), axis=1)
        collision_loss = self.w1 * jnp.exp(-self.w2 * squared_distances)
        collision_loss = jnp.sum(collision_loss) / other_xts.shape[0]

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
        self.device = jax.devices(device)[0]
        self.goal_threshold = goal_threshold
        self.agent_trajectories = None
        self.optimization_iters = optimization_iters 
        self.step_size = step_size
        self.agents: List[Agent] = []
        self.setup_sim(init_ps, goals)
        
        # Create batched jitted functions for better GPU utilization
        self._setup_batched_functions()

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
    
    def _setup_batched_functions(self):
        """Setup batched jitted functions for better GPU utilization."""
        # Create a dummy agent to extract the methods we need to batch
        dummy_agent = self.agents[0]
        
        # Define batched functions that work on arrays of agent data
        def batched_linearize_dyn(x0s, u_trajs):
            """Batched version of linearize_dyn for all agents."""
            def single_agent_linearize_dyn(x0, u_traj):
                return dummy_agent.linearize_dyn(x0, u_traj)
            return vmap(single_agent_linearize_dyn)(x0s, u_trajs)
        
        def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs):
            """Batched version of linearize_loss for all agents."""
            def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj):
                return dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj)
            return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs)
        
        def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
            """Batched version of solve for all agents."""
            def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
                return dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
            return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
        
        def batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs):
            """Batched version of loss for all agents."""
            def single_agent_loss(x_traj, u_traj, ref_traj, other_traj):
                return dummy_agent.loss(x_traj, u_traj, ref_traj, other_traj)
            return vmap(single_agent_loss)(x_trajs, u_trajs, ref_trajs, other_trajs)
        
        # JIT compile the batched functions
        self.jit_batched_linearize_dyn = jit(batched_linearize_dyn, device=self.device)
        self.jit_batched_linearize_loss = jit(batched_linearize_loss, device=self.device)
        self.jit_batched_solve = jit(batched_solve, device=self.device)
        self.jit_batched_loss = jit(batched_loss, device=self.device)
    
    def get_other_player_trajectories(self, agentId: int) -> List[jnp.ndarray]:
        return [agent.x_traj[:, :2] for agent in self.agents if agent.id != agentId]

    def step(self) -> None:
        # Prepare batched inputs for all agents
        x0s = jnp.stack([agent.x0 for agent in self.agents])
        u_trajs = jnp.stack([agent.u_traj for agent in self.agents])
        ref_trajs = jnp.stack([agent.ref_traj for agent in self.agents])
        
        # Step 1: Batched linearize dynamics for all agents
        x_trajs, A_trajs, B_trajs = self.jit_batched_linearize_dyn(x0s, u_trajs)
        
        # Update agent trajectories
        for i, agent in enumerate(self.agents):
            agent.x_traj = x_trajs[i]
            agent.A_traj = A_trajs[i]
            agent.B_traj = B_trajs[i]
        
        # Step 2: Prepare other player trajectories for each agent
        # Create a 3D array where other_trajs[i] contains trajectories of all other agents for agent i
        other_trajs_list = []
        for i in range(self.n_agents):
            other_agent_trajs = [self.agents[j].x_traj[:, :2] for j in range(self.n_agents) if j != i]
            if other_agent_trajs:
                stacked_other = jnp.stack(other_agent_trajs)
                other_trajs = jnp.transpose(stacked_other, (1, 0, 2))  # (horizon, n_others, 2)
            else:
                # If no other agents, create empty array with correct shape
                other_trajs = jnp.zeros((self.horizon, 0, 2))
            other_trajs_list.append(other_trajs)
        
        other_trajs_batch = jnp.stack(other_trajs_list)
        
        # Step 3: Batched linearize loss and solve for all agents
        a_trajs, b_trajs = self.jit_batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs_batch)
        loss_vals = self.jit_batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs_batch)
        v_trajs, _ = self.jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        
        # Update agent states
        for i, agent in enumerate(self.agents):
            agent.a_traj = a_trajs[i]
            agent.b_traj = b_trajs[i]
            agent.loss_history.append(loss_vals[i])
            agent.v_traj = v_trajs[i]
        
        # Step 4: Update control trajectories
        for i, agent in enumerate(self.agents):
            agent.u_traj += self.step_size * v_trajs[i]
        
        # Print logging for loss values
        if int(self.timestep / self.dt) % int(self.optimization_iters/10) == 0:
            loss_str = f"iter[{int(self.timestep / self.dt)}/{self.optimization_iters}] | "
            for i, agent in enumerate(self.agents):
                loss_str += f"Agent {agent.id}: {loss_vals[i]:.3f} | "
            print(loss_str)

        self.timestep += self.dt
    
    def run(self) -> None:
        for _ in tqdm(range(self.optimization_iters)):
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
    init_arena_range = (-5.0, 5.0)  # Initial position range
    goal_threshold = 0.2  # Convergence threshold
    device = "cuda"
    optimization_iters = 250  # Total simulation steps
    step_size = 0.002

    # Cost weights (position, velocity, control)
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # Higher position weights
    R = jnp.diag(jnp.array([0.01, 0.01]))
    W = jnp.array([100.0, 5.0, 0.1, 1.0]) # w1 = collision, w2 = collision cost exp decay, w3 = control, w4 = navigation

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
    