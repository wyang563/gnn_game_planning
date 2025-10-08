import jax
import jax.nn as jnn
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, PRNGKeyArray

class MLP(eqx.Module):
    """
    Multi-Layer Perceptron for processing past player trajectories using Equinox.
    
    Input: Flattened trajectories of shape [batch, n_agents * mask_horizon * 4] = [batch, 400]
           or unbatched [400]
    Output: Per-agent predictions of shape [batch, n_agents] = [batch, 10] with values in range [0, 1]
            or unbatched [10]
    
    Architecture:
    - Input layer: 400 -> 256 (ReLU)
    - Hidden layer 1: 256 -> 64 (ReLU) 
    - Hidden layer 2: 64 -> 16 (ReLU)
    - Output layer: 16 -> 10 (Sigmoid)
    """
    
    layers: list
    
    def __init__(self, n_agents: int, mask_horizon: int, state_dim: int, hidden_sizes: Tuple[int, ...], key: PRNGKeyArray):
        self.mask_horizon = mask_horizon
        self.state_dim = state_dim
        input_dim = (n_agents - 1) * mask_horizon * state_dim
        output_dim = n_agents - 1
        
        # Split the key for each layer initialization
        keys = jax.random.split(key, 4)
        
        # Create layers
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_sizes[0], key=keys[0]),
            eqx.nn.Linear(hidden_sizes[0], hidden_sizes[1], key=keys[1]),
            eqx.nn.Linear(hidden_sizes[1], hidden_sizes[2], key=keys[2]),
            eqx.nn.Linear(hidden_sizes[2], output_dim, key=keys[3])
        ]
    
    def _forward_single(self, x: Array) -> Array:
        """Forward pass for a single unbatched input."""
        # normalize values relative to origin of ego agent last position
        ego_last_position = x[self.mask_horizon * self.state_dim - 4:self.mask_horizon * self.state_dim - 2]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnn.relu(x)
            else:
                x = jnn.sigmoid(x)
        return x
    
    def __call__(self, x: Array) -> Array:
        """Forward pass that handles both batched and unbatched inputs."""
        # Check if input is batched (2D) or unbatched (1D)
        if x.ndim == 1:
            # Unbatched input: shape (input_dim,)
            return self._forward_single(x)
        else:
            # Batched input: shape (batch, input_dim)
            # Use vmap to vectorize over the batch dimension
            return jax.vmap(self._forward_single)(x)
