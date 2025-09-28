# This file is used for inference 
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List, Dict, Any
from utils.utils import origin_init_collision, random_init, load_config, parse_arguments
from utils.point_agent_lqr_plots import LQRPlotter
from tqdm import tqdm
from models.policies import *
from agent import Agent

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

class Simulator:
    def __init__(
        self, 
        n_agents: int,
        Q: jnp.ndarray,
        R: jnp.ndarray, 
        W: jnp.ndarray,
        time_steps: int, # number of horizons calculated per iteration
        horizon: int, 
        mask_horizon: int,
        dt: float, 
        init_arena_range: Tuple[float, float], 
        device: str = "cpu",
        goal_threshold: float = 0.1,
        optimization_iters: int = 1000,
        step_size: float = 0.002,
        init_ps: List[jnp.ndarray] = None,
        goals: List[jnp.ndarray] = None,
        init_type: str = "random",
        limit_past_horizon: bool = False,
        masking_method: str = None,
        top_k: int = 3,
        critical_radius: float = 1.0,
        debug: bool = False,
    ) -> None: 

        self.n_agents = n_agents
        self.Q = Q
        self.R = R
        self.W = W
        self.horizon = horizon
        self.mask_horizon = mask_horizon
        self.time_steps = time_steps
        self.dt = dt
        self.init_arena_range = init_arena_range
        self.device = jax.devices(device)[0]
        self.goal_threshold = goal_threshold
        self.agent_trajectories = None
        self.optimization_iters = optimization_iters 
        self.step_size = step_size
        self.debug = debug
        self.limit_past_horizon = limit_past_horizon
        self.top_k = top_k
        self.critical_radius = critical_radius
        self.masking_method = masking_method

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

        mask_diag = ~jnp.eye(self.n_agents, dtype=bool)
        mask = mask_diag.astype(jnp.int32)
        self.other_index = mask
        
        # metrics we log
        self.all_loss_vals = []
        self.all_min_pairwise_distances = []
        self.player_masks = []
    
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
        
        def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
            """Batched version of linearize_loss for all agents."""
            def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask):
                return dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask)
            return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
        
        def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
            """Batched version of solve for all agents."""
            def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
                return dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
            return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
        
        def batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
            """Batched version of loss for all agents."""
            def single_agent_loss(x_traj, u_traj, ref_traj, other_traj, mask):
                return dummy_agent.loss(x_traj, u_traj, ref_traj, other_traj, mask)
            return vmap(single_agent_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
        
        # JIT compile the batched functions
        self.jit_batched_linearize_dyn = jit(batched_linearize_dyn, device=self.device)
        self.jit_batched_linearize_loss = jit(batched_linearize_loss, device=self.device)
        self.jit_batched_solve = jit(batched_solve, device=self.device)
        self.jit_batched_loss = jit(batched_loss, device=self.device)
    
    def calculate_x_trajs(self):
        x_trajs, _, _ = self.jit_batched_linearize_dyn(self.x0s, self.u_trajs)
        return x_trajs

    def setup_horizon_arrays(self, iter_timestep: int):
        # calculate x0s, u_trajs, ref_trajs starting at a given timestep 
        # to prepare for horizon optimization calculations 
        x_trajs = self.calculate_x_trajs()
        self.horizon_x0s = x_trajs[:, iter_timestep]
        start_ind = iter_timestep 
        end_ind = min(start_ind + self.horizon, self.time_steps)
        self.horizon_u_trajs = jnp.zeros((self.n_agents, end_ind - start_ind, 2))
        self.horizon_rej_trajs = self.ref_trajs[:, start_ind:end_ind, :]
    
    def get_past_x_trajs(self, iter_timestep: int):
        # get all x_traj data from [iter_timestep - horizon, iter_timestep - 1], pad if the 
        # data ends up being shorter than length horizon
        x_trajs = self.calculate_x_trajs()
        if not self.limit_past_horizon:
            start_ind = 0
        else:
            start_ind = max(0, iter_timestep - self.mask_horizon)

        end_ind = iter_timestep
        traj_slice = x_trajs[:, start_ind:end_ind, :]
        
        if not self.limit_past_horizon:
            pad_length = self.time_steps - (end_ind - start_ind)
        else:
            pad_length = self.mask_horizon - (end_ind - start_ind)

        if pad_length > 0:
            padding = jnp.tile(x_trajs[:, start_ind:start_ind+1, :], (1, pad_length, 1))
            traj_slice = jnp.concatenate([padding, traj_slice], axis=1)
        return jax.device_put(traj_slice, self.device)

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

    def run_masking_method(self, masking_method: str, iter_timestep: int):
        past_x_trajs = self.get_past_x_trajs(iter_timestep)
        if masking_method == "nearest_neighbors_top_k":
            self.other_index = nearest_neighbors_top_k(past_x_trajs, top_k=self.top_k)
        elif masking_method == "jacobian_top_k":
            self.other_index = jacobian_top_k(past_x_trajs, top_k=self.top_k, dt=self.dt, w1=self.W[0], w2=self.W[1])
        elif masking_method == "nearest_neighbors_radius":
            self.other_index = nearest_neighbors_radius(past_x_trajs, critical_radius=self.critical_radius)
        elif masking_method == "None":
            pass
        else:
            raise ValueError(f"Invalid masking method: {masking_method}")

    def step(self) -> None:
        # Step 1: Batched linearize dynamics for all agents
        x_trajs, A_trajs, B_trajs = self.jit_batched_linearize_dyn(self.horizon_x0s, self.horizon_u_trajs)
        
        # Step 2: Prepare other player trajectories for each agent
        # Create a 3D array where other_trajs[i] contains trajectories of all other agents for agent i
        # Use actual horizon length from x_trajs shape, not self.horizon
        actual_horizon = x_trajs.shape[1]
        all_x_pos = jnp.broadcast_to(x_trajs[:, :, :2], (self.n_agents, self.n_agents, actual_horizon, 2))
        other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))

        # Step 3: solve for all agents
        # Slice mask to match actual horizon length
        mask_for_step = jnp.tile(self.other_index[:, None, :], (1, actual_horizon, 1))
        a_trajs, b_trajs = self.jit_batched_linearize_loss(x_trajs, self.horizon_u_trajs, self.horizon_rej_trajs, other_x_trajs, mask_for_step)
        v_trajs, _ = self.jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        
        # Step 4: Update control trajectories
        self.horizon_u_trajs += self.step_size * v_trajs

    def run_test(self) -> None:
        for iter_timestep in tqdm(range(self.time_steps)):
            self.setup_horizon_arrays(iter_timestep)

            # calculate mask of other agents to consider for each agent
            self.run_masking_method(self.masking_method, iter_timestep)

            # run optimization for horizon trajectory
            for _ in range(self.optimization_iters):
                self.step()

            if self.debug:
                # calculate loss vals/min pairwise distances
                self.player_masks.append(self.other_index)
                x_trajs, _, _ = self.jit_batched_linearize_dyn(self.x0s, self.u_trajs)
                all_x_pos = x_trajs[:, :, :2]

                min_pairwise_distance = self.global_min_pairwise_distance(all_x_pos, iter_timestep)
                self.all_min_pairwise_distances.append(min_pairwise_distance)

                # calculate loss values for specific time step
                x_trajs_t = x_trajs[:, iter_timestep:iter_timestep+1, :]
                u_trajs_t = self.u_trajs[:, iter_timestep:iter_timestep+1, :]
                ref_trajs_t = self.ref_trajs[:, iter_timestep:iter_timestep+1, :]
                
                # Create other_x_trajs: for each agent, broadcast all agent positions at current timestep
                # Shape should be (n_agents, 1, n_agents, 2) for batched loss calculation
                current_positions = all_x_pos[:, iter_timestep, :2]  # Shape: (n_agents, 2)
                other_trajs_t = jnp.broadcast_to(current_positions[None, None, :, :], (self.n_agents, 1, self.n_agents, 2))
                
                # Create masks: (n_agents, 1, n_agents) - exclude self-interaction
                masks_t = ~jnp.eye(self.n_agents, dtype=bool)
                masks_t = masks_t.astype(jnp.int32)
                masks_t = jnp.expand_dims(masks_t, axis=1)  # Shape: (n_agents, 1, n_agents)

                loss_vals_t = self.jit_batched_loss(x_trajs_t, u_trajs_t, ref_trajs_t, other_trajs_t, masks_t)
                self.all_loss_vals.append(loss_vals_t)
                if iter_timestep % int(self.time_steps/10) == 0:
                    loss_str = f"iter[{iter_timestep}/{self.time_steps}] | "
                    for i, agent in enumerate(self.agents):
                        loss_str += f"Agent {agent.id}: {loss_vals_t[i]:.3f} | "
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
    # DEBUG MODE
    DEBUG = True

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
    
    # extracting masking method config
    masking_config = config['masking']
    limit_past_horizon = masking_config['limit_past_horizon']
    masking_method= masking_config['masking_method']
    top_k = masking_config['top_k']
    critical_radius = masking_config['critical_radius']
    mask_horizon = masking_config['mask_horizon']

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
        mask_horizon=mask_horizon,
        time_steps=time_steps,
        dt=dt,
        init_arena_range=init_arena_range,
        device=device,
        step_size=step_size,
        goal_threshold=goal_threshold,
        optimization_iters=optimization_iters,
        init_type=init_type,
        limit_past_horizon=limit_past_horizon,
        masking_method=masking_method,
        top_k=top_k,
        critical_radius=critical_radius,
        debug=DEBUG,
    )

    # Print initial conditions
    print("Initial conditions:")
    for i, agent in enumerate(simulator.agents):
        print(f"  Agent {i}: pos=({agent.x0[0]:.2f}, {agent.x0[1]:.2f}), goal=({agent.goal[0]:.2f}, {agent.goal[1]:.2f})")
    print()

    # Run simulation with progress updates
    print("Running simulation...")
    simulator.run_test()
    
    # Generate plots
    print("\nGenerating visualization plots...")
    # Derive arena bounds from init_arena_range to ensure everything is in frame
    x_min, x_max = init_arena_range
    # Use same range for y for a square arena assumption
    arena_bounds = (x_min, x_max, x_min, x_max)
    plotter = LQRPlotter(simulator.agents, arena_bounds=arena_bounds)
    
    # Generate all static plots and optionally create GIF
    # Set create_gif=True to generate trajectory animation
    plotter.plot_all(create_gif=True, gif_interval=50, dump_data=True, simulator=simulator)
    
    # Generate ego agent perspective GIFs
    print("\nGenerating ego agent perspective GIFs...")
    plotter.create_ego_agent_gif(simulator, timestep_interval=20, interval=100)
    