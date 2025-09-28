# this file adds onto the Simulator class to train models
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List, Dict, Any
from lqrax import iLQR
from utils.utils import origin_init_collision, random_init, load_config, parse_arguments
from utils.point_agent_lqr_plots import LQRPlotter
from tqdm import tqdm
from models.policies import *
from sim_solver import Simulator
from models.mlp import MLP

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

class SimulatorTrain(Simulator):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
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
        self.epochs = kwargs["epochs"]
        self.model = self._setup_model(model_type=kwargs["model_type"])

    def _setup_model(self, model_type: str):
        if model_type == "mlp":
            return MLP(
                n_agents=self.n_agents,
                mask_horizon=self.mask_horizon,
                state_dim=self.state_dim,
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported")

    def train(self):
        pass