#!/usr/bin/env python3

# Set JAX environment variables to help with GPU issues
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pickle
from tqdm import tqdm
import os
from datetime import datetime
import gc
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard as tb

# JAX/Flax imports
import flax.linen as nn
import optax
from flax.training import train_state
import flax.serialization

# Import from the main lqrax module
import sys
import os
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from load_config import load_config
from data.ref_traj_data_loading import load_reference_trajectories, prepare_batch_for_training, extract_true_goals_from_batch
from models.loss_funcs import binary_loss, mask_sparsity_loss, batch_similarity_loss

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
config = load_config()

# Game parameters
N_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id
dt = config.game.dt
T_total = config.game.T_total
T_observation = config.psn.observation_length
T_reference = config.game.T_total
state_dim = config.game.state_dim
control_dim = config.game.control_dim

# GNN training parameters
num_epochs = config.gnn.num_epochs
learning_rate = config.gnn.learning_rate
batch_size = config.gnn.batch_size
sigma1 = config.gnn.sigma1  
sigma2 = config.gnn.sigma2  
num_message_passing_rounds = config.gnn.num_message_passing_rounds
edge_metric = config.gnn.edge_metric
dropout_rate = config.gnn.dropout_rate

gru_hidden_size = config.gnn.gru_hidden_size
message_mlp_dims = config.gnn.message_mlp_dims
influence_head_dims = config.gnn.influence_head_dims

# Game solving parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size
Q = jnp.diag(jnp.array(config.optimization.Q))  # State cost weights [x, y, vx, vy]
R = jnp.diag(jnp.array(config.optimization.R))               # Control cost weights [ax, ay]

# Reference trajectory parameters - use training data directory for training
# Use dataset directory for loading individual sample files
# This directory contains ref_traj_sample_*.json files
reference_dir = config.training.data_dir

# Device selection - Use config preference
if config.device.preferred_device == "gpu":
    gpu_devices = jax.devices("gpu")
    if gpu_devices:
        device = gpu_devices[0]
        print(f"Using GPU: {device}")
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        print("JAX platform forced to: gpu")
        
        # Test GPU functionality with a simple operation
        test_array = jax.random.normal(jax.random.PRNGKey(config.training.seed), (10, 10))
        test_result = jnp.linalg.inv(test_array)
        print("GPU matrix operations working correctly")
    else:
        if config.device.preferred_device == "gpu":
            raise RuntimeError("No GPU devices found, but GPU is preferred")
        else:
            print("No GPU devices found, falling back to CPU")
            device = jax.devices("cpu")[0]
            print(f"Using CPU: {device}")
else:
    device = jax.devices("cpu")[0]
    print(f"Using CPU: {device}")

# ===================================================
# GNN NETWORK DEFINITIONS
# ===================================================

class MessageMLP(nn.Module):
    hidden_dims: List[int]
    dropout_rate: float = 0.3
    out_dim: int = gru_hidden_size

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f'message_mlp_{i}')(x)
            x = nn.relu(x)
            if i < len(self.hidden_dims) - 1:  # Don't apply dropout to last layer
                x = nn.Dropout(rate=self.dropout_rate, name=f'message_mlp_dropout_{i}')(x, deterministic=deterministic)
        x = nn.Dense(features=self.out_dim, name='message_mlp_output')(x)
        return x

class InfluenceHead(nn.Module):
    hidden_dims: List[int]
    dropout_rate: float = 0.3
    out_dim: int = N_agents - 1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f'edge_mask_mlp_{i}')(x)
            x = nn.relu(x)
            if i < len(self.hidden_dims) - 1:  # Don't apply dropout to last layer
                x = nn.Dropout(rate=self.dropout_rate, name=f'edge_mask_mlp_dropout_{i}')(x, deterministic=deterministic)
        x = nn.Dense(features=self.out_dim, name='edge_mask_mlp_output')(x)
        mask = nn.sigmoid(x)
        return mask

class GNNSelectionNetwork(nn.Module):
    """
    GNN Selection Network for selecting important agents.
    """
    hidden_dims: List[int]
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    obs_input_type: str = "full"    # "full" or "partial"
    mask_output_dim: int = N_agents - 1
    deterministic: bool = True
    num_message_passing_rounds: int = num_message_passing_rounds

    def encode_gru_features(self, x):
        agent_features = []

        for agent_idx in range(N_agents):
            # Extract trajectory for this agent: (batch_size, T_observation, state_dim)
            agent_traj = x[:, :, agent_idx, :]
            
            # Use simple GRU cell with manual scanning for compatibility
            gru_cell = nn.GRUCell(features=self.gru_hidden_size, name=f'gru_agent_{agent_idx}')
            
            # Initialize hidden state
            init_hidden = jnp.zeros((batch_size, self.gru_hidden_size))
            
            # Process sequence step by step
            hidden = init_hidden
            for t in range(T_observation):
                hidden, _ = gru_cell(hidden, agent_traj[:, t, :])
            
            # Use final hidden state as agent representation
            agent_features.append(hidden)
        
        # stack agent features 
        h_nodes = jnp.stack(agent_features, axis=1) # (B, N, gru_hidden_dim)
        return h_nodes

    def create_graph_adj_matrix(self, x):
        # TODO: this will be updated in the future to reflect usage of CBFs to select nearest neighbors
        n_agents = x.shape[2]
        graph_adj_matrix = ~(jnp.eye(n_agents).astype(jnp.bool_))
        graph_adj_matrix = graph_adj_matrix.astype(jnp.float32)

        # tile graph adj matrix to be (B, N, N)
        graph_adj_matrix = jnp.tile(graph_adj_matrix, (batch_size, 1, 1))
        return graph_adj_matrix
    
    def message_pass(self, node_encodings, graph_adj_matrix):
        # message pass between edges
        nodes_i = jnp.tile(node_encodings[:, :, None, :], (1, 1, N_agents, 1))
        nodes_j = jnp.tile(node_encodings[:, None, :, :], (1, N_agents, 1, 1))
        pair_encodings = jnp.concatenate([nodes_i, nodes_j], axis=-1)

        # edge MLP
        message_mlp = MessageMLP(hidden_dims=message_mlp_dims, dropout_rate=dropout_rate)
        messages = message_mlp(pair_encodings, deterministic=True)  # (B, N, N, message_dim)

        # Mask out messages where there's no edge using jnp.where
        mask = graph_adj_matrix[..., None]  # (B, N, N, 1)
        masked_messages = jnp.where(mask, messages, 0.0)
        masked_messages = jnp.transpose(masked_messages, (0, 2, 1, 3))

        # get indegree of each node
        node_indegrees = jnp.sum(jnp.transpose(graph_adj_matrix, (0, 2, 1)), axis=2)
        
        # aggregate messages for each node
        aggregated_messages = jnp.sum(masked_messages, axis=2) / node_indegrees[..., None]  # (B, N, message_dim)

        # update node encodings
        node_encodings = node_encodings + aggregated_messages
        return node_encodings

    @nn.compact
    def __call__(self, x):
        if self.obs_input_type == "partial":
            input_dim = 2  # Only position (x, y)
        else:  # "full"
            input_dim = 4  # Full state (x, y, vx, vy)
        
        # Reshape input to (batch_size, T_observation, N_agents, input_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, T_observation, N_agents, input_dim)
        
        # create graph
        graph_adj_matrix = self.create_graph_adj_matrix(x)
        node_encodings = self.encode_gru_features(x) # initial node encodings in the graph

        # message passing
        for _ in range(self.num_message_passing_rounds):
            node_encodings = self.message_pass(node_encodings, graph_adj_matrix)
        
        # get mask values from influence head
        influence_head = InfluenceHead(hidden_dims=influence_head_dims, dropout_rate=dropout_rate)
        # Use the ego agent's encoding to generate the mask
        ego_encoding = node_encodings[:, ego_agent_id, :]  # (B, hidden_dim)
        mask = influence_head(ego_encoding, deterministic=True)  # (B, N_agents - 1)
        return mask

if __name__ == "__main__":
    # Initialize the model
    model = GNNSelectionNetwork(
        hidden_dims=message_mlp_dims, 
        gru_hidden_size=gru_hidden_size, 
        dropout_rate=dropout_rate, 
        obs_input_type="full", 
        mask_output_dim=N_agents - 1, 
        deterministic=True, 
        num_message_passing_rounds=num_message_passing_rounds
    )
    
    # Create dummy input data
    rng = jax.random.PRNGKey(42)
    dummy_input = jax.random.normal(rng, (batch_size, T_observation, N_agents, 4))
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Observation length: {T_observation}")
    print(f"Number of agents: {N_agents}")
    print(f"State dimension: 4")
    
    # Initialize model parameters
    params = model.init(rng, dummy_input)
    print(f"Model initialized successfully")
    print(f"Number of parameters: {sum(x.size for x in jax.tree.leaves(params))}")
    
    # Forward pass
    output = model.apply(params, dummy_input)
    print(f"Forward pass completed")
    print(f"Output shape: {output.shape}")
    print(f"Output type: {type(output)}")




        


