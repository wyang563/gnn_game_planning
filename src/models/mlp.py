import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random
from flax import nnx
from typing import Tuple, Dict, Callable, List

class MLP(nnx.Module):
    """
    Multi-Layer Perceptron for processing past player trajectories using pure JAX.
    
    Input: Flattened trajectories of shape [n_agents * mask_horizon * 4] = [10 * 10 * 4] = [400]
    Output: Per-agent predictions of shape [n_agents] = [10] with values in range [0, 1]
    
    Architecture:
    - Input layer: 400 -> 256 (ReLU)
    - Hidden layer 1: 256 -> 64 (ReLU) 
    - Hidden layer 2: 64 -> 16 (ReLU)
    - Output layer: 16 -> 10 (Sigmoid)
    """
    
    def __init__(self, n_agents: int, mask_horizon: int, state_dim: int, 
                 hidden_sizes: Tuple[int, ...], random_seed: int):
        self.n_agents = n_agents
        self.mask_horizon = mask_horizon
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        self.input_dim = n_agents * mask_horizon * state_dim
        self.output_dim = n_agents
        
        # Initialize parameters
        self.params = self._init_params(random_seed)
    
    def _init_params(self, random_seed: int) -> Dict:
        rngs = nnx.Rngs(random_seed)
        linear1 = nnx.Linear(self.input_dim, self.hidden_sizes[0], rngs=rngs)
        linear2 = nnx.Linear(self.hidden_sizes[0], self.hidden_sizes[1], rngs=rngs)
        linear3 = nnx.Linear(self.hidden_sizes[1], self.hidden_sizes[2], rngs=rngs)
        linear4 = nnx.Linear(self.hidden_sizes[2], self.output_dim, rngs=rngs)
        self.layers = nnx.List([linear1, linear2, linear3, linear4])
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = nnx.relu(layer(x))
            else:
                x = nnx.sigmoid(layer(x))
        return x
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)
