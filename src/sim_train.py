import jax
import jax.numpy as jnp
import jax.random
from jax import jit, vmap, grad, value_and_grad
import optax
import equinox as eqx
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
        self.step_size = kwargs.get("step_size", 0.002)
        self.dt = kwargs.get("dt", 0.1)
        
        # Cost matrices
        self.Q = jnp.diag(jnp.array(kwargs.get("Q", [1.0, 1.0, 0.1, 0.1])))
        self.R = jnp.diag(jnp.array(kwargs.get("R", [0.1, 0.1])))
        self.W = jnp.array(kwargs.get("W", [5.0, 1.0, 0.1, 1.0]))
        
        self.sigma_1 = kwargs["sigma_1"]
        self.sigma_2 = kwargs["sigma_2"]
        self.optimization_iters = kwargs["optimization_iters"]

        # setup dataset
        self.dataloader = self._setup_dataloader(
            model_type=kwargs["model_type"],
            dataset_path=kwargs["dataset_dir"],
        )
        
        # Setup batched agent functions for optimization
        self._setup_batched_agent_functions()

    def _setup_dataloader(
        self,
        model_type: str,
        dataset_path: str,
    ):
        if model_type == "mlp":
            dataset = create_mlp_dataloader(
                inputs_path=os.path.join(dataset_path, "inputs.zarr"),
                targets_path=os.path.join(dataset_path, "targets.zarr"),
                x0s_path=os.path.join(dataset_path, "x0s.zarr"),
                ref_trajs_path=os.path.join(dataset_path, "ref_trajs.zarr"),
                shuffle_buffer=500,
            )
            return dataset.create_jax_iterator()
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
    
    def step(self, x0s, ref_trajs, u_trajs, horizon_size, player_masks):
        x_trajs, A_trajs, B_trajs = self.jit_batched_linearize_dyn(x0s, u_trajs)

        all_x_pos = jnp.broadcast_to(x_trajs[:, :, :2], (self.n_agents, self.n_agents, horizon_size, 2))
        other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))
        
        mask_for_step = jnp.tile(player_masks[:, None, :], (1, horizon_size, 1))
        a_trajs, b_trajs = self.jit_batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_x_trajs, mask_for_step)
        v_trajs, _ = self.jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)

        u_trajs = u_trajs + self.step_size * v_trajs
        return u_trajs
    
    def compute_loss(self, model, inputs, x0s, ref_trajs, targets):
        """Forward pass through model and compute loss."""
        # Get player masks from model
        player_masks = model(inputs)
        
        # Calculate optimal trajectory with current masks
        horizon_size = ref_trajs.shape[1]
        u_trajs = jnp.zeros((self.n_agents, horizon_size, 2))
        
        for _ in range(self.optimization_iters):
            u_trajs = self.step(x0s, ref_trajs, u_trajs, horizon_size, player_masks)
        
        # Compute final trajectories
        x_trajs, _, _ = self.jit_batched_linearize_dyn(x0s, u_trajs)
        
        # Compute loss
        binary_loss = jnp.sum(jnp.abs(0.5 - jnp.abs(0.5 - player_masks)), axis=-1)
        mask_loss = self.sigma_1 * jnp.sum(jnp.abs(player_masks), axis=-1) / player_masks.shape[0]
        x_pos = x_trajs[:, :, :2]
        target_pos = targets[:, :, :2]

        # sim loss
        d2 = jnp.square(x_pos - target_pos).sum(axis=-1)
        sim_loss = self.sigma_2 * jnp.sqrt(d2).sum(axis=-1)
        
        # Sum all losses and compute mean
        total_loss = binary_loss + mask_loss + sim_loss
        return total_loss

    def train(self, model, optimizer, opt_state):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, x0s, ref_trajs, targets in tqdm(self.dataloader, desc="Training"):
                # Squeeze batch dimension
                inputs = inputs.squeeze(0)
                x0s = x0s.squeeze(0)
                ref_trajs = ref_trajs.squeeze(0)
                targets = targets.squeeze(0)

                # Compute loss and gradients using Equinox
                loss, grads = eqx.filter_value_and_grad(self.compute_loss)(model, inputs, x0s, ref_trajs, targets)

                # Update parameters
                updates, opt_state = optimizer.update(grads, opt_state, model)
                model = eqx.apply_updates(model, updates)

                # Accumulate loss for logging
                epoch_loss += float(loss)
                num_batches += 1
            
            # Print epoch statistics
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.epochs} - Average Loss: {avg_loss:.6f}")
        
        return model


def setup_model(
    model_type: str,
    state_dim: int,
    n_agents: int,
    mask_horizon: int,
    random_seed: int = 42,
):
    if model_type == "mlp":
        key = jax.random.PRNGKey(random_seed)
        return MLP(
            n_agents=n_agents,
            mask_horizon=mask_horizon,
            state_dim=state_dim,
            hidden_sizes=(256, 64, 16),  # Default architecture from paper
            key=key
        )
    # TODO: add GNN model later
    else:
        raise ValueError(f"Model type {model_type} not supported")

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
    
    model = setup_model(
        model_type=config["model_type"],
        state_dim=config["state_dim"],
        n_agents=config["n_agents"],
        mask_horizon=config["mask_horizon"],
        random_seed=config["random_seed"],
    )

    optimizer = optax.adam(learning_rate=config["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    trainer = SimulatorTrain(
        model_type=config["model_type"],
        n_agents=config["n_agents"],
        mask_horizon=config["mask_horizon"],
        random_seed=config["random_seed"],
        epochs=config["epochs"],
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

    final_model = trainer.train(model=model, optimizer=optimizer, opt_state=opt_state)
    