# this file adds onto the Simulator class to train models
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

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

class SimulatorTrain:
    def __init__(
        self,
        **kwargs
    ):
        self.epochs = kwargs["epochs"]
        self.learning_rate = kwargs.get("learning_rate", 0.001)
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

    def _setup_model(
        self,
        model_type: str,
        state_dim: int = 4, 
        n_agents: int = 10, 
        mask_horizon: int = 10,
        random_seed: int = 42
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
    
    def _flatten_x_trajs(self, x_trajs):
        n_agents = x_trajs.shape[0]
        mask_horizon = x_trajs.shape[1]
        state_dim = x_trajs.shape[2]
        batch = jnp.broadcast_to(x_trajs, (n_agents, n_agents, mask_horizon, state_dim))
        
        result = batch.copy()
        
        for i in range(n_agents):
            temp = result[i, 0].copy()
            result = result.at[i, 0].set(result[i, i])
            result = result.at[i, i].set(temp)
        
        flattened_batch = result.reshape(n_agents, n_agents * mask_horizon * state_dim)
        return flattened_batch

    def calc_loss(self, model):
        # loop we define to create x_trajs
        x_trajs = self.simulator.calculate_x_trajs()
        self.simulator.setup_horizon_arrays(x_trajs, self.iter_timestep)

        # run model to get mask
        past_x_trajs = self.simulator.get_past_x_trajs(x_trajs, self.iter_timestep)
        flattened_batch = self._flatten_x_trajs(past_x_trajs)
        predicted_masks = model(flattened_batch)
        self.simulator.other_index = predicted_masks
        cached_u_trajs = self.simulator.horizon_u_trajs.copy()

        # calculate trajectory values with model mask applied
        for _ in range(self.simulator.optimization_iters):
            self.simulator.step()

        model_x_trajs, _, _ = self.simulator.jit_batched_linearize_dyn(self.simulator.horizon_x0s, self.simulator.horizon_u_trajs)
        u_traj_update = self.simulator.horizon_u_trajs[:, 0, :]

        # reset simulator horizon_u_trajs
        self.simulator.horizon_u_trajs = cached_u_trajs
        mask_diag = ~jnp.eye(self.simulator.n_agents, dtype=bool)
        self.simulator.other_index = mask_diag.astype(jnp.int32)

        # calculate loss values with all agents
        for _ in range(self.simulator.optimization_iters):
            self.simulator.step()

        target_x_trajs, _, _ = self.simulator.jit_batched_linearize_dyn(self.simulator.horizon_x0s, self.simulator.horizon_u_trajs)

        # update u_trajs
        self.simulator.u_trajs = self.simulator.u_trajs.at[:, self.iter_timestep, :].set(u_traj_update)

        # calculate loss values
        target_positions = target_x_trajs[:, :, :2]
        model_positions = model_x_trajs[:, :, :2]

        binary_loss = jnp.sum(jnp.abs(0.5 - jnp.abs(predicted_masks - 0.5)))
        mask_loss = self.sigma_1 * jnp.sum(jnp.abs(predicted_masks))

        # calculate traj loss
        l2 = jnp.sqrt(jnp.sum(jnp.square(model_positions - target_positions), axis=-1))
        traj_loss = self.sigma_2 * jnp.sum(l2)

        total_loss = binary_loss + mask_loss + traj_loss
        return total_loss, {"binary_loss": binary_loss, "mask_loss": mask_loss, "traj_loss": traj_loss}
    
    def compute_loss_and_grad(self, model): 
        """
        Compute loss and gradients for the model.
        
        Args:
            flattened_batch: Input features for the model
            model_x_trajs: Trajectories from model predictions
            target_x_trajs: Target trajectories (ground truth)
            
        Returns:
            total_loss: Scalar loss value
            loss_dict: Dictionary with individual loss components
            grads: Gradients with respect to model parameters
        """
        def compute_loss(model):
            total_loss, loss_dict = self.calc_loss(model)
            return total_loss, loss_dict
        
        # Compute loss and gradients using nnx.value_and_grad
        (total_loss, loss_dict), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
        
        return total_loss, loss_dict, grads
    
    def update_model(self, grads):
        """
        Update model parameters using computed gradients.
        
        Args:
            grads: Gradients with respect to model parameters (nnx.State object)
        """
        self.optimizer.update(self.model, grads)

    def train(self, **kwargs):
        for epoch in range(self.epochs):
            print(f"Training epoch {epoch}")

            # initiate new instance of simulator
            self.simulator = Simulator(
                n_agents=kwargs["n_agents"],
                Q=kwargs["Q"],
                R=kwargs["R"],
                W=kwargs["W"],
                time_steps=kwargs["time_steps"],
                horizon=kwargs["horizon"],
                mask_horizon=kwargs["mask_horizon"],
                dt=kwargs["dt"],
                init_arena_range=kwargs["init_arena_range"],
                device=kwargs["device"],
                goal_threshold=kwargs["goal_threshold"],
                optimization_iters=kwargs["optimization_iters"],
                step_size=kwargs["step_size"],
                init_type=kwargs["init_type"],
                limit_past_horizon=kwargs["limit_past_horizon"],
                masking_method=kwargs["masking_method"],
                top_k=kwargs["top_k"],
                critical_radius=kwargs["critical_radius"],
                debug=kwargs["debug"],
            )

            # run full loop through simulator
            for iter_timestep in tqdm(range(self.simulator.time_steps)):
                self.iter_timestep = iter_timestep
                total_loss, loss_dict, grads = self.compute_loss_and_grad(self.model)
                self.update_model(grads)
                if iter_timestep % 50 == 0:
                    print(f"Iteration {iter_timestep}, Total Loss: {total_loss}, Binary Loss: {loss_dict['binary_loss']}, Mask Loss: {loss_dict['mask_loss']}, Traj Loss: {loss_dict['traj_loss']}")
                
if __name__ == "__main__":
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
    
    # Extract config parameters and set up SimulatorTrain
    simulator_config = config["simulator"]
    masking_config = config["masking"]
    training_config = config["training"]

    Q = jnp.diag(jnp.array(simulator_config['Q']))  # Higher position weights
    R = jnp.diag(jnp.array(simulator_config['R']))
    W = jnp.array(simulator_config['W'])  # w1 = collision, w2 = collision cost exp decay, w3 = control, w4 = navigation
    
    # Combine all config parameters needed for SimulatorTrain initialization
    train_params = {
        # Training-specific parameters for SimulatorTrain constructor
        "epochs": training_config["epochs"],
        "learning_rate": training_config["learning_rate"],
        "model_type": training_config["model_type"],
        "state_dim": training_config["state_dim"],
        "n_agents": simulator_config["n_agents"],
        "mask_horizon": masking_config["mask_horizon"],
        "random_seed": training_config["random_seed"],
        "sigma_1": training_config["sigma_1"],
        "sigma_2": training_config["sigma_2"],
    }
    
    # Parameters needed for Simulator within the training loop
    simulator_params = {
        # Simulator parameters
        "n_agents": simulator_config["n_agents"],
        "horizon": simulator_config["horizon"],
        "time_steps": simulator_config["time_steps"],
        "dt": simulator_config["dt"],
        "init_arena_range": simulator_config["init_arena_range"],
        "init_type": simulator_config["init_type"],
        "goal_threshold": simulator_config["goal_threshold"],
        "device": simulator_config["device"],
        "optimization_iters": simulator_config["optimization_iters"],
        "step_size": simulator_config["step_size"],
        "W": W,
        "Q": Q,
        "R": R,
        
        # Masking parameters
        "limit_past_horizon": masking_config["limit_past_horizon"],
        "mask_horizon": masking_config["mask_horizon"],
        "masking_method": masking_config["masking_method"],
        "top_k": masking_config["top_k"],
        "critical_radius": masking_config["critical_radius"],
        
        # Debug parameter (not in YAML, set from constant)
        "debug": DEBUG,
    }
    
    # Initialize SimulatorTrain
    print("Initializing SimulatorTrain...")
    trainer = SimulatorTrain(**train_params)
    
    # Start training
    print("Starting training...")
    trainer.train(**simulator_params)

