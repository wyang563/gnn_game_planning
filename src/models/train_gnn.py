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
from data.ref_traj_data_loading import load_reference_trajectories, prepare_batch_for_training_gnn, extract_true_goals_from_batch, sort_by_n_agents, organize_batches
from models.loss_funcs import binary_loss, mask_sparsity_loss, batch_similarity_loss, batch_ego_agent_cost

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================
config = load_config()

# Game parameters
N_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id
dt = config.game.dt
T_total = config.game.T_total
T_observation = config.game.T_observation
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
    def __call__(self, x, deterministic: bool = False):
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
    def __call__(self, x, deterministic: bool = False):
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
    gru_time_decay_factor: float = 0.8
    dropout_rate: float = 0.3
    obs_input_type: str = "full"    # "full" or "partial"
    deterministic: bool = False
    num_message_passing_rounds: int = num_message_passing_rounds

    def gru_encoder_node_features(self, x):
        agent_features = []
        batch_size = x.shape[0]
        n_agents = x.shape[2]
        
        # Use simple GRU cell with manual scanning for compatibility
        node_encoder_gru_cell = nn.GRUCell(features=self.gru_hidden_size, name=f'node_gru_shared')

        for agent_idx in range(n_agents):
            # Extract trajectory for this agent: (batch_size, T_observation, state_dim)
            agent_traj = x[:, :, agent_idx, :]
            
            # Initialize hidden state
            init_hidden = jnp.zeros((batch_size, self.gru_hidden_size))
            
            # Process sequence step by step
            hidden = init_hidden
            for t in range(T_observation):
                discount_factor = self.gru_time_decay_factor ** (T_observation - 1 - t)
                hidden, _ = node_encoder_gru_cell(hidden, discount_factor * agent_traj[:, t, :])
            
            # Use final hidden state as agent representation
            agent_features.append(hidden)
        
        # stack agent features 
        h_nodes = jnp.stack(agent_features, axis=1) # (B, N, gru_hidden_dim)
        return h_nodes
    
    # functions for calculating edge features
    def gru_encoder_edge_features(self, batch_size, n_agents, x_traj_delta, gru_encoder):
        edge_features = [[None for _ in range(n_agents)] for _ in range(n_agents)]
        for agent_idx_i in range(n_agents):
            for agent_idx_j in range(n_agents):
                if agent_idx_i != agent_idx_j:
                    agent_traj_delta = x_traj_delta[:, :, agent_idx_i, agent_idx_j, :]
                    init_hidden = jnp.zeros((batch_size, self.gru_hidden_size))
                    hidden = init_hidden
                    for t in range(T_observation):
                        discount_factor = self.gru_time_decay_factor ** (T_observation - 1 - t)
                        hidden, _ = gru_encoder(hidden, discount_factor * agent_traj_delta[:, t, :])
                    edge_features[agent_idx_i][agent_idx_j] = hidden
        edge_features = jnp.array(edge_features)
        return edge_features

    def calculate_closing_speed(self, x_traj_delta):
        dp = x_traj_delta[..., :2]
        dv = x_traj_delta[..., 2:]
        num = -jnp.sum(dp * dv, axis=-1)             
        den = jnp.linalg.norm(dp, axis=-1) + 1e-5    
        return num / den

    def calculate_edge_features(self, x): 
        # calculate pairwise differences between all agents for trajectories
        batch_size = x.shape[0]
        n_agents = x.shape[2]

        x_i = x[:, :, :, None, :]  # (B, T, N, 1, input_dim)
        x_j = x[:, :, None, :, :]  # (B, T, 1, N, input_dim)
        x_traj_delta = x_j - x_i   # (B, T, N, N, input_dim)
        x_traj_delta_last = x_traj_delta[:, -1, :, :, :] 

        # define gru feature encoders
        x_traj_delta_gru_encoder = nn.GRUCell(features=self.gru_hidden_size, name=f'edge_x_traj_delta_gru_shared')
        closing_speeds_gru_encoder = nn.GRUCell(features=self.gru_hidden_size, name=f'edge_closing_speeds_gru_shared')

        # encode edge features in time series
        gru_encoded_x_delta_featuers = self.gru_encoder_edge_features(batch_size, n_agents, x_traj_delta, x_traj_delta_gru_encoder)
        closing_speeds = self.calculate_closing_speed(x_traj_delta_last)
        gru_encoded_closing_speeds = self.gru_encoder_edge_features(batch_size, n_agents, closing_speeds, closing_speeds_gru_encoder)

        # concatenate all calculated edge features to get final edge encodings
        edge_features = jnp.concatenate([gru_encoded_x_delta_featuers, gru_encoded_closing_speeds], axis=-1)
        return edge_features

    def create_graph_adj_matrix(self, x):
        # TODO: this will be updated in the future to reflect usage of CBFs to select nearest neighbors
        n_agents = x.shape[2]
        batch_size = x.shape[0]
        
        # Create a random graph adj matrix for each sample in the batch
        graph_adj_matrix = jax.random.randint(jax.random.PRNGKey(42), (batch_size, n_agents, n_agents), 0, 2)
        graph_adj_matrix = graph_adj_matrix.astype(jnp.float32)
        
        return graph_adj_matrix
    
    def message_pass(self, node_encodings, graph_adj_matrix, message_mlp, deterministic=False):
        n_agents = graph_adj_matrix.shape[1]
        # message pass between edges
        nodes_i = jnp.tile(node_encodings[:, :, None, :], (1, 1, n_agents, 1))
        nodes_j = jnp.tile(node_encodings[:, None, :, :], (1, n_agents, 1, 1))
        pair_encodings = jnp.concatenate([nodes_i, nodes_j], axis=-1)

        # edge MLP
        messages = message_mlp(pair_encodings, deterministic=deterministic)  # (B, N, N, message_dim)

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
    def __call__(self, x, rngs=None, deterministic=False):
        # assume input is (batch_size, T_observation, N_agents, input_dim)
        graph_adj_matrix = self.create_graph_adj_matrix(x)
        node_encodings = self.gru_encoder_node_features(x) # initial node encodings in the graph
        edge_encodings = self.calculate_edge_features(x) # initial edge encodings in the graph
        message_mlp = MessageMLP(hidden_dims=message_mlp_dims, dropout_rate=dropout_rate)

        # message passing
        for _ in range(self.num_message_passing_rounds):
            node_encodings = self.message_pass(node_encodings, graph_adj_matrix, message_mlp, deterministic)
        
        # get mask values corresponding to each edge from influence head
        influence_head = InfluenceHead(hidden_dims=influence_head_dims, dropout_rate=dropout_rate)

        # Use the ego agent's encoding to generate the mask
        ego_encoding = node_encodings[:, ego_agent_id, :]  # (B, hidden_dim)
        transposed_graph_adj_matrix = jnp.transpose(graph_adj_matrix, (0, 2, 1))
        batch_ego_incoming_edge_indices = transposed_graph_adj_matrix[:, ego_agent_id, :] == 1.0
        ego_encoding_repeated = jnp.tile(ego_encoding[:, None, :], (1, node_encodings.shape[1], 1))
        influence_head_inputs = jnp.concatenate([node_encodings, ego_encoding_repeated], axis=-1)
        influence_outputs = influence_head(influence_head_inputs, deterministic=deterministic)
        influence_outputs = jnp.squeeze(influence_outputs, axis=-1)

        # zero out influence outputs for non-incoming edges
        output_masks = jnp.where(batch_ego_incoming_edge_indices, influence_outputs, 0.0)
        return output_masks

# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================
def load_trained_gnn_models(gnn_model_path: Optional[str], obs_input_type: str = "full") -> Tuple[Optional[GNNSelectionNetwork], Any]:
    """
    Load trained GNNSelectionNetwork model from files.
    """
    if gnn_model_path is not None:
        print(f"Loading trained GNNSelectionNetwork model from: {gnn_model_path}")
        with open(gnn_model_path, "rb") as f:
            gnn_model_bytes = pickle.load(f)
        gnn_model = GNNSelectionNetwork(hidden_dims=message_mlp_dims, gru_hidden_size=gru_hidden_size, dropout_rate=dropout_rate, obs_input_type=obs_input_type)
        gnn_trained_state = flax.serialization.from_bytes(gnn_model, gnn_model_bytes)
        print("✓ GNNSelectionNetwork model loaded successfully")
    else:
        print("No GNNSelectionNetwork model provided (using baseline method)")
        gnn_model, gnn_trained_state = None, None
    return gnn_model, gnn_trained_state

# ===================================================
# TRAINING FUNCTIONS 
# ===================================================

def create_train_state(model: nn.Module, optimizer: optax.GradientTransformation, 
                      input_shape: Tuple[int, ...], rng: jnp.ndarray) -> train_state.TrainState:
    """Create training state for the model."""
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input, deterministic=False)
    params = variables['params']
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state

def train_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    batch_data: List[Dict[str, Any]],
    sigma1: float,
    sigma2: float,
    rng: jnp.ndarray = None,
    loss_type: str = "similarity",
    obs_input_type: str = "full"
) -> Tuple[train_state.TrainState, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    observations, reference_trajectories = batch

    def loss_fn_similarity(params):
        predicted_masks = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        predicted_goals = extract_true_goals_from_batch(batch_data)
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        similarity_loss_val = batch_similarity_loss(predicted_masks, predicted_goals, batch_data, obs_input_type=obs_input_type)
        total_loss = similarity_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val
        return total_loss, (binary_loss_val, sparsity_loss_val, similarity_loss_val)
    
    def loss_fn_ego_cost(params):
        predicted_masks = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        predicted_goals = extract_true_goals_from_batch(batch_data)
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        ego_cost_loss_val = batch_ego_agent_cost(predicted_masks, predicted_goals, batch_data, obs_input_type=obs_input_type, apply_masks=False)
        total_loss = ego_cost_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val
        return total_loss, (binary_loss_val, sparsity_loss_val, ego_cost_loss_val)
    
    if loss_type == "similarity":
        (loss, loss_components), grads = jax.value_and_grad(loss_fn_similarity, has_aux=True)(state.params)
    elif loss_type == "ego_agent_cost":
        (loss, loss_components), grads = jax.value_and_grad(loss_fn_ego_cost, has_aux=True)(state.params)
    else:
        raise ValueError(f"loss type invalid: {loss_type}")

    # Apply gradient clipping to prevent gradient explosion
    grad_norm = jax.tree.reduce(lambda x, y: x + jnp.sum(jnp.square(y)), grads, initializer=0.0)
    grad_norm = jnp.sqrt(grad_norm)
    
    max_grad_norm = config.debug.gradient_clip_value
    if grad_norm > max_grad_norm:
        scale = max_grad_norm / grad_norm
        grads = jax.tree.map(lambda g: g * scale, grads)
    
    # Apply gradients using the optimizer
    state = state.apply_gradients(grads=grads)
    
    return state, loss, loss_components

def validation_step(state: train_state.TrainState, validation_data: List[Dict[str, Any]],
                   batch_size: int = 32, ego_agent_id: int = 0, obs_input_type: str = "full", loss_type: str = "similarity") -> Tuple[float, float, float, float, List[jnp.ndarray]]:
    """
    Perform validation on the validation dataset.
    
    Args:
        state: Current train state
        validation_data: Validation dataset
        goal_model: Pretrained goal inference network
        goal_trained_state: Trained state of goal inference network
        batch_size: Batch size for validation
        ego_agent_id: ID of the ego agent
        
    Returns:
        Tuple of (average validation loss, average binary loss, average sparsity loss, average similarity loss, list of predicted masks)
    """
    val_losses = []
    val_binary_losses = []
    val_sparsity_losses = []
    val_similarity_losses = []
    all_predicted_masks = []
    
    # Create validation batches
    num_val_batches = (len(validation_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_val_batches):
        batch_data = validation_data[batch_idx]
        
        # Prepare batch for validation
        observations, reference_trajectories = prepare_batch_for_training_gnn(batch_data, obs_input_type)
        
        # Get predicted mask from PSN (validation mode, no dropout)
        predicted_masks = state.apply_fn({'params': state.params}, observations, deterministic=True)
        
        # Store masks for analysis
        all_predicted_masks.append(predicted_masks)
        
        # Extract true goals from reference data
        predicted_goals = extract_true_goals_from_batch(batch_data)
        
        # Compute validation loss components
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        
        # Compute similarity loss using game solving
        if loss_type == "similarity":
            similarity_loss_val = batch_similarity_loss(predicted_masks, predicted_goals, batch_data)
        elif loss_type == "ego_agent_cost":
            similarity_loss_val = batch_ego_agent_cost(predicted_masks, predicted_goals, batch_data, obs_input_type=obs_input_type, apply_masks=False)
        else:
            raise ValueError(f"loss type invalid: {loss_type}")
        
        # Total validation loss (same as training loss for fair comparison)
        total_loss_val = similarity_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val

        val_losses.append(float(total_loss_val))
        val_binary_losses.append(float(binary_loss_val))
        val_sparsity_losses.append(float(sparsity_loss_val))
        val_similarity_losses.append(float(similarity_loss_val))
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_binary = sum(val_binary_losses) / len(val_binary_losses)
    avg_val_sparsity = sum(val_sparsity_losses) / len(val_sparsity_losses)
    avg_val_similarity = sum(val_similarity_losses) / len(val_similarity_losses)

    return avg_val_loss, avg_val_binary, avg_val_sparsity, avg_val_similarity, all_predicted_masks

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
    num_message_passing_rounds: int,
    loss_type: str,
    rng: jnp.ndarray,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], 
           List[float], List[float], List[float], train_state.TrainState, str, float, int]:
    
    # setup log directories
    config_name = f"gnn_{obs_input_type}_planning_true_goals_maxN_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{learning_rate}_bs_{batch_size}_sigma1_{sigma1}_sigma2_{sigma2}_epochs_{num_epochs}_mp_{num_message_passing_rounds}_loss_type_{loss_type}"
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
    validation_data_batches = organize_batches(validation_data_by_n_agents)

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
            observations, reference_trajectories = prepare_batch_for_training_gnn(batch_data, obs_input_type)
            rng, step_key = jax.random.split(rng)
            state, loss, (binary_loss_val, sparsity_loss_val, similarity_loss_val) = train_step(
                state,
                (observations, reference_trajectories),
                batch_data,
                sigma1,
                sigma2,
                step_key,
                loss_type,
                obs_input_type
            )

            epoch_losses.append(float(loss))
            epoch_binary_losses.append(float(binary_loss_val))
            epoch_sparsity_losses.append(float(sparsity_loss_val))
            epoch_similarity_losses.append(float(similarity_loss_val))

            # Update batch progress bar
            batch_pbar.set_postfix({
                'Loss': f'{float(loss):.4f}',
                'Binary': f'{float(binary_loss_val):.4f}',
                'Sparsity': f'{float(sparsity_loss_val):.4f}',
                'Similarity': f'{float(similarity_loss_val):.4f}'
            })
            batch_pbar.update(1)

            # Update main training progress bar
            training_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Batch': f'{batch_idx+1}/{num_batches}',
                'Loss': f'{float(loss):.4f}',
            })
            training_pbar.update(1)
    
        # Close batch progress bar
        batch_pbar.close()

        # validation step
        val_loss, val_binary_loss, val_sparsity_loss, val_similarity_loss, val_masks = validation_step(
            state,
            validation_data_batches,
            batch_size,
            ego_agent_id=ego_agent_id,
            loss_type=loss_type,
            obs_input_type=obs_input_type,
        )
        validation_losses.append(val_loss)
        validation_binary_losses.append(val_binary_loss)
        validation_sparsity_losses.append(val_sparsity_loss)
        validation_similarity_losses.append(val_similarity_loss)

        # Calculate average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_binary_loss = sum(epoch_binary_losses) / len(epoch_binary_losses)
        avg_sparsity_loss = sum(epoch_sparsity_losses) / len(epoch_sparsity_losses)
        avg_similarity_loss = sum(epoch_similarity_losses) / len(epoch_similarity_losses)
        training_losses.append(avg_loss)
        binary_losses.append(avg_binary_loss)
        sparsity_losses.append(avg_sparsity_loss)
        similarity_losses.append(avg_similarity_loss)

        # Track best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            
            # Save best model using proper JAX/Flax serialization
            best_model_path = os.path.join(run_log_dir, "psn_best_model.pkl")
            model_bytes = flax.serialization.to_bytes(state)
            with open(best_model_path, 'wb') as f:
                pickle.dump(model_bytes, f)
            print(f"\nNew best model found at epoch {epoch + 1} with validation loss: {best_loss:.4f}")
            print(f"Best model saved to: {best_model_path}")
        
        # Save model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(run_log_dir, f"psn_checkpoint_epoch_{epoch + 1}.pkl")
            model_bytes = flax.serialization.to_bytes(state)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model_bytes, f)
            print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")
        
        # Update main progress bar
        training_pbar.set_postfix({
            'Epoch': f'{epoch+1}/{num_epochs}',
            'Train Loss': f'{avg_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
        })
        
        # Log epoch-level metrics to TensorBoard
        writer.add_scalar('Loss/Epoch/Training', avg_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Epoch/Best', best_loss, epoch)
        writer.add_scalar('Loss/Epoch/Binary', avg_binary_loss, epoch)
        writer.add_scalar('Loss/Epoch/Sparsity', avg_sparsity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Similarity', avg_similarity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation_Binary', val_binary_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation_Sparsity', val_sparsity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation_Similarity', val_similarity_loss, epoch)
        writer.add_scalar('Training/EpochProgress', (epoch + 1) / num_epochs, epoch)
        
        # Training progress metrics
        writer.add_scalar('Progress/EpochProgress', (epoch + 1) / num_epochs, epoch)
        writer.add_scalar('Progress/LossImprovement', float(best_loss - val_loss), epoch)

        # Add text summary for hyperparameters
        if epoch == 0:
            writer.add_text('Hyperparameters', f"""
            - Number of agents: {N_agents}
            - Total game steps: {T_total}
            - Observation steps: {T_observation}
            - Learning rate: {learning_rate}
            - Batch size: {batch_size}
            - Sigma1: {sigma1}
            - Sigma2: {sigma2}
            - Total epochs: {num_epochs}
            """, epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val: {best_loss:.4f} (epoch {best_epoch+1})")
        
        # Clean up memory after each epoch
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    
    # Close progress bar
    training_pbar.close()
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed! Best model found at epoch {best_epoch + 1} with loss: {best_loss:.4f}")
    
    return training_losses, validation_losses, binary_losses, sparsity_losses, similarity_losses, validation_binary_losses, validation_sparsity_losses, validation_similarity_losses, state, log_dir, best_loss, best_epoch

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
    loss_type = config.gnn.loss_type
    
    print(f"GNN model created with observation type: {config.gnn.obs_input_type}")
    
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
        obs_input_type=config.psn.obs_input_type,
        loss_type=loss_type,
        num_message_passing_rounds=num_message_passing_rounds,
    )







