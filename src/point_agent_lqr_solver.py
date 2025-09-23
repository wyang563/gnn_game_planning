import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from utils.utils import origin_init_collision, random_init, load_config, parse_arguments
from utils.point_agent_lqr_plots import LQRPlotter
from tqdm import tqdm

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")


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
        time_steps: int,
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
        self.time_steps = time_steps
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
        self.ref_traj = jax.device_put(self.get_ref_traj(), self.device)
        self.goal_threshold = goal_threshold
        self.x_traj: Optional[jnp.ndarray] = None  

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

    # whole simulation versions of function
    def get_ref_traj(self):
        return jnp.linspace(self.x0[:2], self.goal, self.time_steps+1)[1:]
    
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
        time_steps: int, # number of horizons calculated per iteration
        horizon: int, 
        dt: float, 
        init_arena_range: Tuple[float, float], 
        device: str = "cpu",
        goal_threshold: float = 0.1,
        optimization_iters: int = 1000,
        step_size: float = 0.002,
        init_ps: List[jnp.ndarray] = None,
        goals: List[jnp.ndarray] = None,
        init_type: str = "random",
    ) -> None: 

        self.n_agents = n_agents
        self.Q = Q
        self.R = R
        self.W = W
        self.horizon = horizon
        self.time_steps = time_steps
        self.dt = dt
        self.init_arena_range = init_arena_range
        self.device = jax.devices(device)[0]
        self.goal_threshold = goal_threshold
        self.agent_trajectories = None
        self.optimization_iters = optimization_iters 
        self.step_size = step_size
        self.agents: List[Agent] = []
        self.setup_sim(init_ps, goals, init_type)
        
        # Create batched jitted functions for better GPU utilization
        self._setup_batched_functions()

    def setup_sim(self, init_ps: List[jnp.ndarray] = None, goals: List[jnp.ndarray] = None, init_type: str = "random") -> None:
        if init_type == "origin":
            init_ps, goals = origin_init_collision(self.n_agents, self.init_arena_range)
        elif init_type == "random":
            init_ps, goals = random_init(self.n_agents, self.init_arena_range)

        for i in range(self.n_agents):
            px, py = float(init_ps[i][0]), float(init_ps[i][1]) 
            x0 = jnp.array([px, py, -0.8, 0.0])
            u_traj = jnp.zeros((self.time_steps, 2))
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
                time_steps=self.time_steps,
                x0=x0,
                u_traj=u_traj,
                goal=goal_i,
                device=self.device,
                goal_threshold=self.goal_threshold
            )

            self.agents.append(agent)
        
        # setup batched dynamics matrices for all agents
        self.x0s = jnp.stack([agent.x0 for agent in self.agents])
        self.u_trajs = jnp.stack([agent.u_traj for agent in self.agents])
        self.ref_trajs = jnp.stack([agent.ref_traj for agent in self.agents])
        
        all_idx = jnp.arange(self.n_agents)
        self.other_index = jnp.stack([
            jnp.concatenate([all_idx[:i], all_idx[i+1:]])
            for i in range(self.n_agents)
        ])
        
        # metrics we log
        self.all_loss_vals = []
        self.all_min_pairwise_distances = []
    
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
    
    def setup_horizon_arrays(self, iter_timestep: int):
        # calculate x0s, u_trajs, ref_trajs starting at a given timestep 
        # to prepare for horizon calculations
        x_trajs, _, _ = self.jit_batched_linearize_dyn(self.x0s, self.u_trajs)
        self.horizon_x0s = x_trajs[:, iter_timestep]
        start_ind = iter_timestep 
        end_ind = min(start_ind + self.horizon, self.time_steps)
        self.horizon_u_trajs = jnp.zeros((self.n_agents, end_ind - start_ind, 2))
        self.horizon_rej_trajs = self.ref_trajs[:, start_ind:end_ind, :]

    def global_min_pairwise_distance(self, all_x_pos: jnp.ndarray, iter_timestep: int) -> jnp.ndarray:
        """Return min distance between any pair of distinct agents at a given timestep.

        Args:
            all_x_pos: Array of shape (num_agents, num_timesteps, 2)
            iter_timestep: Timestep index at which to compute pairwise distance
        """
        # Positions at the specified timestep: shape (num_agents, 2)
        X_t = all_x_pos[:, iter_timestep, :]
        N = X_t.shape[0]

        # Compute pairwise squared distances at this timestep only: shape (N, N)
        d2 = jnp.sum((X_t[:, None, :] - X_t[None, :, :]) ** 2, axis=-1)

        # Exclude self-distances
        mask = ~jnp.eye(N, dtype=bool)
        masked_d2 = jnp.where(mask, d2, jnp.inf)

        min_d2 = jnp.min(masked_d2)
        return jnp.sqrt(min_d2)

    def step(self) -> None:
        # Step 1: Batched linearize dynamics for all agents
        x_trajs, A_trajs, B_trajs = self.jit_batched_linearize_dyn(self.horizon_x0s, self.horizon_u_trajs)
        
        # Step 2: Prepare other player trajectories for each agent
        # Create a 3D array where other_trajs[i] contains trajectories of all other agents for agent i
        all_x_pos = x_trajs[:, :, :2]
        other_x_pos = jnp.take(all_x_pos, self.other_index, axis=0)
        other_trajs_batch = jnp.transpose(other_x_pos, (0, 2, 1, 3))

        # Step 3: solve for all agents
        a_trajs, b_trajs = self.jit_batched_linearize_loss(x_trajs, self.horizon_u_trajs, self.horizon_rej_trajs, other_trajs_batch)
        v_trajs, _ = self.jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        
        # Step 4: Update control trajectories
        self.horizon_u_trajs += self.step_size * v_trajs
    
    def run(self) -> None:
        for iter_timestep in tqdm(range(self.time_steps)):
            self.setup_horizon_arrays(iter_timestep)
            
            # run optimization for horizon trajectory
            for _ in range(self.optimization_iters):
                self.step()

            # calculate loss vals/min pairwise distances
            x_trajs, _, _ = self.jit_batched_linearize_dyn(self.x0s, self.u_trajs)
            all_x_pos = x_trajs[:, :, :2]
            other_x_pos = jnp.take(all_x_pos, self.other_index, axis=0)
            other_trajs_batch = jnp.transpose(other_x_pos, (0, 2, 1, 3))
            loss_vals = self.jit_batched_loss(x_trajs, self.u_trajs, self.ref_trajs, other_trajs_batch)
            self.all_loss_vals.append(loss_vals)
            min_pairwise_distance = self.global_min_pairwise_distance(all_x_pos, iter_timestep)
            self.all_min_pairwise_distances.append(min_pairwise_distance)
            if iter_timestep % int(self.time_steps/10) == 0:
                loss_str = f"iter[{iter_timestep}/{self.time_steps}] | "
                for i, agent in enumerate(self.agents):
                    loss_str += f"Agent {agent.id}: {loss_vals[i]:.3f} | "
                print(loss_str)

            # update u_trajs
            self.u_trajs = self.u_trajs.at[:, iter_timestep, :].set(self.horizon_u_trajs[:, 0, :])
        
        # calculate x_traj/u_traj for all agents based off batched values in simulator
        for i, agent in enumerate(self.agents):
            agent.u_traj = self.u_trajs[i, :, :]
            agent.x_traj = agent.calculate_x_traj()

        # check convergence
        for agent in self.agents:
            print(f"Agent {agent.id}, converged={agent.check_convergence()}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration from YAML file
    try:
        config = load_config(args.config)
        print(f"=== Testing Receding Horizon Nash Game Simulator ===")
        print(f"Using config file: {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Extract simulation parameters from config
    simulator_config = config['simulator']
    N = simulator_config['n_agents']  # Number of agents
    horizon = simulator_config['horizon']  # Optimization horizon (T steps)
    dt = simulator_config['dt']  # Time step
    init_arena_range = tuple(simulator_config['init_arena_range'])  # Initial position range
    goal_threshold = simulator_config['goal_threshold']  # Convergence threshold
    device = simulator_config['device']
    optimization_iters = simulator_config['optimization_iters']  # Total simulation steps
    step_size = simulator_config['step_size']
    init_type = simulator_config['init_type']

    # Cost weights (position, velocity, control)
    Q = jnp.diag(jnp.array(simulator_config['Q']))  # Higher position weights
    R = jnp.diag(jnp.array(simulator_config['R']))
    W = jnp.array(simulator_config['W'])  # w1 = collision, w2 = collision cost exp decay, w3 = control, w4 = navigation
    time_steps = int(simulator_config['time_steps'])

    # Create simulator    
    simulator = Simulator(
        n_agents=N,
        Q=Q,
        R=R,
        W=W,
        horizon=horizon,
        time_steps=time_steps,
        dt=dt,
        init_arena_range=init_arena_range,
        device=device,
        step_size=step_size,
        goal_threshold=goal_threshold,
        optimization_iters=optimization_iters,
        init_type=init_type,
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
    plotter.plot_all(create_gif=True, gif_interval=50, dump_data=False, simulator=simulator)
    