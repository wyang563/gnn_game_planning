import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from typing import Tuple, Optional
from jaxtyping import Array, PRNGKeyArray
import matplotlib.pyplot as plt
import numpy as np

class MLP(eqx.Module):
    """
    Multi-Layer Perceptron for processing past player trajectories using Equinox.
    
    Input: Flattened trajectories of shape [batch, (n_agents - 1) * mask_horizon * 4] = [batch, 360]
           or unbatched [400]
    Output: Per-agent predictions of shape [batch, n_agents] = [batch, 10] with values in range [0, 1]
            or unbatched [10]
    
    Architecture:
    - Input layer: 400 -> 256 (ReLU)
    - Hidden layer 1: 256 -> 64 (ReLU) 
    - Hidden layer 2: 64 -> 16 (ReLU)
    - Output layer: 16 -> 9 (Sigmoid)
    """
    
    layers: list
    mask_horizon: int
    state_dim: int
    mode: str
    
    def __init__(self, n_agents: int, mask_horizon: int, state_dim: int, hidden_sizes: Tuple[int, ...], key: PRNGKeyArray, mode: str = "test"):
        self.mask_horizon = mask_horizon
        self.state_dim = state_dim
        self.mode = mode # either test or train
        input_dim = n_agents * mask_horizon * state_dim
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

    def _standardize_inputs(self, x: Array) -> Array:
        ego_last_position = x[0, -1, :2]  
        ego_last_velocity = x[0, -1, 2:]
        centered_positions = x[:, :, :2] - ego_last_position  
        centered_velocities = x[:, :, 2:] - ego_last_velocity  

        standardized = jnp.concatenate([centered_positions, centered_velocities], axis=-1)  
        return standardized.reshape(-1)  

    def _forward_single(self, x: Array) -> Array:
        """Forward pass for a single unbatched input."""
        x = self._standardize_inputs(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnn.relu(x)
            else:
                x = jnn.sigmoid(x)
        return x
    
    def __call__(self, x: Array) -> Array: 
        """Forward pass that handles both batched and unbatched inputs.
        
        Args:
            x: Input array (batched or unbatched)
            key: Optional PRNG key for data augmentation during training
        """
        # Check if input is batched (2D) or unbatched (1D)
        # visualize centered positions
        if x.ndim == 1:
            # Unbatched input: shape (input_dim,)
            return self._forward_single(x)
        else:
            # Batched input: shape (batch, input_dim)
            return jax.vmap(lambda sample: self._forward_single(sample))(x)
    
    def visualize_centered_positions(self, x: Array):
        """
        Visualize the centered positions from the input data.
        
        Args:
            x: Input array of shape [num_agents, mask_horizon, state_dim]
            save_path: Optional path to save the plot
            title: Title for the plot
        """
        
        # Extract centered positions
        ego_last_position = x[0, -1, :2]  
        centered_positions = x[:, :, :2] - ego_last_position
        centered_positions = np.asarray(centered_positions)
        _, ax = plt.subplots(1, 1, figsize=(14, 10))
        for i in range(centered_positions.shape[0]):
            color = f'C{i}'
            past_x = centered_positions[i, :, 0]
            past_y = centered_positions[i, :, 1]
            ax.plot(past_x, past_y, ':', alpha=0.7, linewidth=2, color=color, 
                    label=f'Agent {i}')

        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title("Centered Past Positions", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig("plots/centered_past_positions.png", dpi=300, bbox_inches='tight')