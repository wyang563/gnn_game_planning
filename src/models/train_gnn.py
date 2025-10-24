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
import random
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from load_config import load_config
from data.ref_traj_data_loading import load_reference_trajectories, prepare_batch_for_training, extract_true_goals_from_batch, sort_by_n_agents, organize_batches
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
    out_dim: int = 1

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
    deterministic: bool = True
    num_message_passing_rounds: int = num_message_passing_rounds

    def encode_gru_features(self, x):
        agent_features = []
        n_agents = x.shape[2]

        for agent_idx in range(n_agents):
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
        batch_size = x.shape[0]
        
        # Create a random graph adj matrix for each sample in the batch
        graph_adj_matrix = jax.random.randint(jax.random.PRNGKey(42), (batch_size, n_agents, n_agents), 0, 2)
        graph_adj_matrix = graph_adj_matrix.astype(jnp.float32)
        
        return graph_adj_matrix
    
    def message_pass(self, node_encodings, graph_adj_matrix, message_mlp):
        n_agents = graph_adj_matrix.shape[1]
        # message pass between edges
        nodes_i = jnp.tile(node_encodings[:, :, None, :], (1, 1, n_agents, 1))
        nodes_j = jnp.tile(node_encodings[:, None, :, :], (1, n_agents, 1, 1))
        pair_encodings = jnp.concatenate([nodes_i, nodes_j], axis=-1)

        # edge MLP
        messages = message_mlp(pair_encodings, deterministic=self.deterministic)  # (B, N, N, message_dim)

        # Mask out messages where there's no edge using jnp.where
        mask = graph_adj_matrix[..., None]  # (B, N, N, 1)
        masked_messages = jnp.where(mask, messages, 0.0)
        masked_messages = jnp.transpose(masked_messages, (0, 2, 1, 3))

        # get indegree of each node
        node_indegrees = jnp.sum(jnp.transpose(graph_adj_matrix, (0, 2, 1)), axis=2)
        node_indegrees = jnp.where(node_indegrees != 0, node_indegrees, 1e-5) # avoid division by zero
        
        # aggregate messages for each node
        aggregated_messages = jnp.sum(masked_messages, axis=2) / node_indegrees[..., None]  # (B, N, message_dim)

        # update node encodings
        node_encodings = node_encodings + aggregated_messages
        return node_encodings

    @nn.compact
    def __call__(self, x):
        # assume input is (batch_size, T_observation, N_agents, input_dim)
        # create graph
        graph_adj_matrix = self.create_graph_adj_matrix(x)
        node_encodings = self.encode_gru_features(x) # initial node encodings in the graph
        message_mlp = MessageMLP(hidden_dims=message_mlp_dims, dropout_rate=dropout_rate)

        # message passing
        for _ in range(self.num_message_passing_rounds):
            node_encodings = self.message_pass(node_encodings, graph_adj_matrix, message_mlp)
        
        # get mask values corresponding to each edge from influence head
        influence_head = InfluenceHead(hidden_dims=influence_head_dims, dropout_rate=dropout_rate)

        # Use the ego agent's encoding to generate the mask
        ego_encoding = node_encodings[:, ego_agent_id, :]  # (B, hidden_dim)
        transposed_graph_adj_matrix = jnp.transpose(graph_adj_matrix, (0, 2, 1))
        batch_ego_incoming_edge_indices = transposed_graph_adj_matrix[:, ego_agent_id, :] == 1.0
        ego_encoding_repeated = jnp.tile(ego_encoding[:, None, :], (1, node_encodings.shape[1], 1))
        influence_head_inputs = jnp.concatenate([node_encodings, ego_encoding_repeated], axis=-1)
        influence_outputs = influence_head(influence_head_inputs, deterministic=self.deterministic)
        influence_outputs = jnp.squeeze(influence_outputs, axis=-1)

        # zero out influence outputs for non-incoming edges
        output_masks = jnp.where(batch_ego_incoming_edge_indices, influence_outputs, 0.0)
        return output_masks

# ===================================================
# TRAINING FUNCTIONS 
# ===================================================

def create_train_state(model: nn.Module, optimizer: optax.GradientTransformation, 
                      input_shape: Tuple[int, ...], rng: jnp.ndarray) -> train_state.TrainState:
    """Create training state for the model."""
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state

def train_gnn(
    gnn_model: nn.Module,
    training_data: List[Dict[str, Any]],
    validation_data: List[Dict[str, Any]],
    num_epochs: int,
    learning_rate: float,
    sigma1: float,
    sigma2: float,
    batch_size: int,
    obs_input_type: str,
    rng: jnp.ndarray,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], 
           List[float], List[float], List[float], train_state.TrainState, str, float, int]:
    
    # setup log directories
    config_name = f"gnn_{obs_input_type}_planning_true_goals_maxN_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{learning_rate}_bs_{batch_size}_sigma1_{sigma1}_sigma2_{sigma2}_epochs_{num_epochs}"
    model_log_dir = os.path.join("log", config_name)
    os.makedirs(model_log_dir, exist_ok=True)
    print(f"This GNN model type for training logs will be saved under: {model_log_dir}")

    # write data to specific run log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(model_log_dir, timestamp)
    os.makedirs(run_log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = tb.SummaryWriter(run_log_dir)
    
    # Create optimizer with weight decay (AdamW)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=5e-4  # L2 regularization to prevent overfitting
    )

    # Determine observation dimensions based on input type
    if obs_input_type == "partial":
        obs_dim = 2  # Only position (x, y)
    else:  # "full"
        obs_dim = 4  # Full state (x, y, vx, vy)

    # Create train state
    input_shape = (batch_size, T_observation, N_agents, obs_dim)
    state = create_train_state(gnn_model, optimizer, input_shape, rng)

    training_losses = []
    validation_losses = []
    # Track individual loss components over epochs
    binary_losses = []
    sparsity_losses = []
    similarity_losses = []
    validation_binary_losses = []
    validation_sparsity_losses = []
    validation_similarity_losses = []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0

    # Main training loop
    print(f"Starting PSN training with pretrained goals...")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Loss weights: σ1={sigma1}, σ2={sigma2}")
    print(f"Training data: {len(training_data)} samples")
    print(f"Validation data: {len(validation_data)} samples")
    print(f"Device: {jax.devices()[0]}")
    print("-" * 80)

    # Main training progress bar
    total_steps = num_epochs * ((len(training_data) + batch_size - 1) // batch_size)
    training_pbar = tqdm(total=total_steps, desc="Training Progress", position=0)

    training_data_by_n_agents = sort_by_n_agents(training_data)
    validation_data_by_n_agents = sort_by_n_agents(validation_data)

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_binary_losses = []
        epoch_sparsity_losses = []
        epoch_similarity_losses = []

        # setup data batches
        training_data_batches = organize_batches(training_data_by_n_agents)
        random.shuffle(training_data_batches)

        num_batches = len(training_data_batches)

        batch_pbar = tqdm(range(num_batches), 
                         desc=f"Epoch {epoch+1}/{num_epochs} - Batches", 
                   position=1, leave=False)
        
        for batch_idx in range(num_batches):
            batch_data = training_data_batches[batch_idx]
            observations, reference_trajectories = prepare_batch_for_training(batch_data, obs_input_type)
            rng, step_key = jax.random.split(rng)



if __name__ == "__main__":
    print("=" * 80)
    print("GNN Training") 
    print("=" * 80)

    # Initialize the model
    gnn_model = GNNSelectionNetwork(
        hidden_dims=message_mlp_dims, 
        gru_hidden_size=gru_hidden_size, 
        dropout_rate=dropout_rate, 
        obs_input_type="full", 
        deterministic=True, 
        num_message_passing_rounds=num_message_passing_rounds
    )
    
    # Create dummy input data
    rng = jax.random.PRNGKey(config.training.seed)

    # Load reference trajectories
    reference_dir = os.path.join("src/data", config.training.gnn_data_dir)
    print(f"Loading reference trajectories from directory: {reference_dir}")
    training_data, validation_data = load_reference_trajectories(reference_dir)
    
    print(f"GNN model created with observation type: {config.gnn.obs_input_type}")

    # Test the GNN model with dummy input
    print("Testing GNN model with dummy input...")
    
    # Create dummy input with shape (batch_size, T_observation, N_agents, state_dim)
    dummy_input = jax.random.normal(rng, (batch_size, T_observation, N_agents, state_dim))
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Initialize the model parameters
    model_params = gnn_model.init(rng, dummy_input)
    print(f"Model parameters initialized successfully")
    
    # Apply the model
    output = gnn_model.apply(model_params, dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Model output (first sample): {output[0]}")
    
    training_losses, validation_losses, binary_losses, sparsity_losses, \
    ego_agent_costs, validation_binary_losses, validation_sparsity_losses, \
    validation_ego_agent_costs, trained_state, log_dir, best_loss, best_epoch = train_gnn(
        gnn_model,
        training_data,
        validation_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        sigma1=sigma1,
        sigma2=sigma2,
        batch_size=batch_size,
        rng=rng,
        obs_input_type=config.psn.obs_input_type
    )






