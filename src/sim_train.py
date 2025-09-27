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
from agent import Agent
from sim_solver import Simulator

# Configure JAX to use float32 by default for better performance
jax.config.update("jax_default_dtype_bits", "32")

class SimulatorTrain(Simulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        pass