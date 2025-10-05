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

class SimulatorTrain:
    def __init__(self, **kwargs):
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

        # setup dataset
        self.dataloader = self._setup_dataloader(
            model_type=kwargs["model_type"],
            dataset_path=kwargs["dataset_dir"],
            batch_size=kwargs["batch_size"]
        )

    def _setup_dataloader(
        self,
        model_type: str,
        dataset_path: str,
        batch_size: int,
    ):
        if model_type == "mlp":
            return create_mlp_dataloader(
                inputs_path=os.path.join(dataset_path, "inputs.zarr"),
                targets_path=os.path.join(dataset_path, "targets.zarr"),
                x0s_path=os.path.join(dataset_path, "x0s.zarr"),
                ref_trajs_path=os.path.join(dataset_path, "ref_trajs.zarr"),
                batch_size=batch_size,
                shuffle_buffer=500,
            )
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
    
    def train(self):
        pass

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
    )
    