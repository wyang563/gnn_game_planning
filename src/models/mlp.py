import jax
import jax.numpy as jnp
import jax.nn as jnn
# import equinox as eqx
from typing import Tuple, Optional, List
from jaxtyping import Array, PRNGKeyArray
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn

class MLP(nn.Module):
    """
    Player Selection Network (PSN) that learns to select important agents and infer their goals.
    
    Input: First 10 steps of all agents' trajectories (T_observation * N_agents * state_dim)
    Output: 
        - Binary mask for selecting other agents (excluding ego agent)
        - Goal positions for all agents (including ego agent)
    """
    
    hidden_dims: List[int]
    n_agents: int = 10
    mask_horizon: int = 10
    state_dim: int = 4
    mask_output_dim: int = n_agents - 1  # Mask for other agents (excluding ego agent)
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass of PSN.
        
        Args:
            x: Input tensor of shape (batch_size, T_observation * N_agents * state_dim)
               Flattened observation trajectories
            
        Returns:
            tuple: (mask, goals) where:
                - mask: Binary mask of shape (batch_size, N_agents - 1)
                - goals: Goal positions of shape (batch_size, N_agents * 2)
        """
        # Reshape input to (batch_size, T_observation, N_agents, state_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.mask_horizon, self.n_agents, self.state_dim)
        
        # Average over time steps to get a summary representation
        x = jnp.mean(x, axis=1)  # (batch_size, N_agents, state_dim)
        
        # Flatten all agent states
        x = x.reshape(batch_size, self.n_agents * self.state_dim)
        
        # Shared feature extraction layers
        x = nn.Dense(features=self.hidden_dims[0])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dims[1])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dims[2])(x)
        x = nn.relu(x)
        mask = nn.Dense(features=self.mask_output_dim)(x)
        mask = nn.sigmoid(mask)  # Binary mask
        return mask
    
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