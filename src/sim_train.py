import jax
import jax.numpy as jnp
import jax.random
from jax import jit, vmap, grad, value_and_grad
import optax
from flax import nnx
from flax.training.train_state import TrainState
from utils.utils import load_config, parse_arguments
from tqdm import tqdm
from models.policies import *
from sim_solver import Simulator
from models.mlp import MLP
from data.mlp_dataset import MLPDataset, create_mlp_dataloader
import os
from agent import Agent

# Configure JAX to use CPU if CUDA has issues
try:
    _ = jax.devices("cuda")
except:
    jax.config.update("jax_platform_name", "cpu")

class SimulatorTrain:
    def __init__(self, **kwargs):
        self.epochs = kwargs["epochs"]
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.n_agents = kwargs["n_agents"]
        self.mask_horizon = kwargs["mask_horizon"]
        self.state_dim = kwargs["state_dim"]
        self.step_size = kwargs.get("step_size", 0.002)
        self.dt = kwargs.get("dt", 0.1)
        
        # Cost matrices
        self.Q = jnp.diag(jnp.array(kwargs.get("Q", [1.0, 1.0, 0.1, 0.1])))
        self.R = jnp.diag(jnp.array(kwargs.get("R", [0.1, 0.1])))
        self.W = jnp.array(kwargs.get("W", [5.0, 1.0, 0.1, 1.0]))
        
        self.model = self._setup_model(
            model_type=kwargs["model_type"], 
            state_dim=kwargs["state_dim"],
            n_agents=kwargs["n_agents"],
            mask_horizon=kwargs["mask_horizon"],
            random_seed=kwargs["random_seed"]
        )
    
        # Setup optimizer - initialize with model parameters
        tx = optax.adam(self.learning_rate)
        self.optimizer = nnx.Optimizer(self.model, tx, wrt=nnx.Param)
        self.sigma_1 = kwargs["sigma_1"]
        self.sigma_2 = kwargs["sigma_2"]
        self.optimization_iters = kwargs["optimization_iters"]

        # setup dataset
        self.dataloader = self._setup_dataloader(
            model_type=kwargs["model_type"],
            dataset_path=kwargs["dataset_dir"],
            batch_size=kwargs["batch_size"]
        )
        
        # Setup batched agent functions for optimization
        self._setup_batched_agent_functions()

    def _setup_dataloader(
        self,
        model_type: str,
        dataset_path: str,
        batch_size: int,
    ):
        if model_type == "mlp":
            dataset = create_mlp_dataloader(
                inputs_path=os.path.join(dataset_path, "inputs.zarr"),
                targets_path=os.path.join(dataset_path, "targets.zarr"),
                x0s_path=os.path.join(dataset_path, "x0s.zarr"),
                ref_trajs_path=os.path.join(dataset_path, "ref_trajs.zarr"),
                batch_size=batch_size,
                shuffle_buffer=500,
            )
            return dataset.create_jax_iterator()
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def _setup_model(
        self,
        model_type: str,
        state_dim: int,
        n_agents: int,
        mask_horizon: int,
        random_seed: int = 42,
    ):
        if model_type == "mlp":
            return MLP(
                n_agents=n_agents,
                mask_horizon=mask_horizon,
                state_dim=state_dim,
                hidden_sizes=(256, 64, 16),  # Default architecture from paper
                random_seed=random_seed
            )
        # TODO: add GNN model later
        else:
            raise ValueError(f"Model type {model_type} not supported")
    
    def _setup_batched_agent_functions(self):
        """Setup batched agent functions for optimization (similar to Simulator)."""
        
        # Create a dummy agent to use its methods for batching
        dummy_x0 = jnp.zeros(4)
        dummy_u_traj = jnp.zeros((self.mask_horizon, 2))
        dummy_goal = jnp.zeros(2)
        
        self.dummy_agent = Agent(
            id=0,
            dt=self.dt,
            x_dim=4,
            u_dim=2,
            Q=self.Q,
            R=self.R,
            W=self.W,
            horizon=self.mask_horizon,
            time_steps=self.mask_horizon,
            x0=dummy_x0,
            u_traj=dummy_u_traj,
            goal=dummy_goal,
        )
        
        # Define batched functions
        def batched_linearize_dyn(x0s, u_trajs):
            """Batched linearization of dynamics."""
            def single_agent_linearize_dyn(x0, u_traj):
                return self.dummy_agent.linearize_dyn(x0, u_traj)
            return vmap(single_agent_linearize_dyn)(x0s, u_trajs)
        
        def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
            """Batched linearization of loss."""
            def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask):
                return self.dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask)
            return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
        
        def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
            """Batched solve for control updates."""
            def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
                return self.dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
            return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
        
        # JIT compile for efficiency (but still differentiable)
        self.jit_batched_linearize_dyn = jit(batched_linearize_dyn)
        self.jit_batched_linearize_loss = jit(batched_linearize_loss)
        self.jit_batched_solve = jit(batched_solve)
    
    def step(self, x0, u_traj, ref_traj, other_ref_trajs, player_mask):
        """
        Perform one optimization step for a single agent considering other agents.
        
        Args:
            x0: Initial state [4]
            u_traj: Control trajectory [horizon, 2]
            ref_traj: Reference trajectory [horizon, 2]
            other_ref_trajs: Other agents' reference trajectories [n_agents, horizon, 2]
            player_mask: Interaction mask [n_agents] - which agents to consider
        
        Returns:
            Updated u_traj after one optimization step
        """
        # Get actual horizon from u_traj shape
        horizon = u_traj.shape[0]
        
        # Step 1: Linearize dynamics for the focal agent
        x_traj, A_traj, B_traj = self.dummy_agent.linearize_dyn(x0, u_traj)
        # x_traj: [horizon, 4]
        
        # Step 2: Use other agents' reference trajectories as their predicted positions
        # other_ref_trajs: [n_agents, horizon, 2]
        # We need [horizon, n_agents, 2] for the loss function
        other_x_trajs = jnp.transpose(other_ref_trajs, (1, 0, 2))
        # other_x_trajs: [horizon, n_agents, 2]
        
        # Step 3: Expand mask to horizon dimension
        # player_mask: [n_agents] -> [horizon, n_agents]
        mask_for_step = jnp.tile(player_mask[None, :], (horizon, 1))
        
        # Step 4: Linearize loss and solve
        a_traj, b_traj = self.dummy_agent.linearize_loss(
            x_traj, u_traj, ref_traj, other_x_trajs, mask_for_step
        )
        v_traj, _ = self.dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
        
        # Step 5: Update control trajectory
        updated_u_traj = u_traj + self.step_size * v_traj
        
        return updated_u_traj    

    def _unflatten_inputs(self, inputs):
        """
        Unflatten inputs to get trajectories for all agents.
        
        Args:
            inputs: [batch_size, 400] where 400 = n_agents * mask_horizon * state_dim
        
        Returns:
            trajs: [batch_size, n_agents, mask_horizon, state_dim]
        """
        batch_size = inputs.shape[0]
        trajs = inputs.reshape(batch_size, self.n_agents, self.mask_horizon, self.state_dim)
        return trajs
    
    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for inputs, x0s, ref_trajs, targets in tqdm(self.dataloader, desc="Training"):
                batch_size = inputs.shape[0]
                horizon = ref_trajs.shape[1]
                player_masks = self.model(inputs)
                
                # Unflatten inputs to get all agents' past trajectories
                # Shape: [batch_size, n_agents, mask_horizon, 4]
                past_trajs = self._unflatten_inputs(inputs)
                
                # Extract other agents' trajectories (excluding focal agent at index 0)
                # For simplicity, assume other agents continue on their last known trajectory
                # Get last position of each other agent and create simple reference trajectories
                other_last_pos = past_trajs[:, 1:, -1, :2]  # [batch_size, n_agents-1, 2]
                
                # Create simple reference trajectories: other agents stay at last position
                # Shape: [batch_size, n_agents-1, horizon, 2]
                other_ref_trajs = jnp.tile(
                    other_last_pos[:, :, None, :], 
                    (1, 1, horizon, 1)
                )
                
                # Add a dummy trajectory at index 0 for the focal agent (not used)
                # Shape: [batch_size, n_agents, horizon, 2]
                other_ref_trajs_full = jnp.concatenate([
                    ref_trajs[:, None, :, :],  # Focal agent ref traj
                    other_ref_trajs
                ], axis=1)
                
                # Initialize control trajectories [batch_size, horizon, 2]
                u_trajs = jnp.zeros((batch_size, horizon, 2))
                
                # Vectorize optimization over batch
                def optimize_single_sample(x0, u_traj, ref_traj, other_refs, mask):
                    for _ in range(self.optimization_iters):
                        u_traj = self.step(x0, u_traj, ref_traj, other_refs, mask)
                    return u_traj
                
                # Run optimization for all samples in batch
                optimized_u_trajs = vmap(optimize_single_sample)(
                    x0s, u_trajs, ref_trajs, other_ref_trajs_full, player_masks
                )
                
                # TODO: Compute loss and backprop through model
                # loss = compute_trajectory_loss(optimized_u_trajs, targets, player_masks)
                # Update model parameters

if __name__ == "__main__":
    DEBUG = False

    args = parse_arguments()
    
    # Load configuration from YAML file
    try:
        config = load_config(args.config)
        print(f"=== Training Receding Horizon Nash Game Simulator ===")
        print(f"Using config file: {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    trainer = SimulatorTrain(
        model_type=config["model_type"],
        state_dim=config["state_dim"],
        n_agents=config["n_agents"],
        mask_horizon=config["mask_horizon"],
        random_seed=config["random_seed"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        dataset_dir=config["dataset_dir"],
        sigma_1=config["sigma_1"],
        sigma_2=config["sigma_2"],
        optimization_iters=config["optimization_iters"],
        dt=config["dt"],
        step_size=config["step_size"],
        Q=config["Q"],
        R=config["R"],
        W=config["W"],
    )

    trainer.train()
    