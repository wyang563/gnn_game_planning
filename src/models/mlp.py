import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import optax
from typing import Tuple, Dict, Callable

class MLP(optax.Module):
    def __init__(
        self,
        n_agents: int = 10,
        mask_horizon: int = 10,
        hidden_sizes: Tuple[int, ...] = (100, 100),
        seed: int = 42,
    ):
        pass
        