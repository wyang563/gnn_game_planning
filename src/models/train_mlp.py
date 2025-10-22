#!/usr/bin/env python3
"""
PSN Training with Pretrained Goal Inference Network

This script trains the Player Selection Network (PSN) using a pretrained
goal inference network. The PSN learns to select important agents while
the goal inference network provides accurate goal predictions.

Stage 2 of the two-stage training approach:
1. Stage 1: Pretrain Goal Inference Network (completed)
2. Stage 2: Train PSN using pretrained goals (this script)

Log Organization:
- PSN training logs are organized under the specific goal inference model directory
- Structure: log/goal_inference_*/psn_pretrained_goals_*/
- This allows easy comparison between different goal inference models

DEBUG MODE:
- Using small dataset for faster debugging
- Game solving is preserved - masks affect game dynamics
- Focus on fixing gradient flow in the game-solving pipeline

Author: Assistant
Date: 2024
"""

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
from lqrax import iLQR  # Now needed for PointAgent
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

# PSN training parameters
num_epochs = config.psn.num_epochs
learning_rate = config.psn.learning_rate
batch_size = config.psn.batch_size
sigma1 = config.psn.sigma1  # Final mask sparsity weight (will gradually increase from 0)
sigma2 = config.psn.sigma2  # Binary loss weight   

# Game solving parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size
Q = jnp.diag(jnp.array(config.optimization.Q))  # State cost weights [x, y, vx, vy]
R = jnp.diag(jnp.array(config.optimization.R))               # Control cost weights [ax, ay]

# Reference trajectory parameters - use training data directory for training
# Use dataset directory for loading individual sample files
# This directory contains ref_traj_sample_*.json files
reference_dir = config.training.data_dir

# hidden dims
psn_hidden_dims = config.get(f"psn.hidden_dims_{N_agents}p")

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

# ============================================================================
# PSN NETWORK DEFINITIONS (GOAL-FREE VERSION)
# ============================================================================
class PlayerSelectionNetwork(nn.Module):
    """
    Player Selection Network (PSN) using GRU for temporal sequence processing.
    
    Input: First 10 steps of all agents' trajectories 
        - If obs_input_type="full": (T_observation * N_agents * 4)
        - If obs_input_type="partial": (T_observation * N_agents * 2)
    Output: Binary mask for selecting other agents (excluding ego agent)
    """
    
    hidden_dims: List[int]
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    mask_output_dim: int = N_agents - 1  # Mask for other agents (excluding ego)
    obs_input_type: str = "full"  # "full" or "partial"
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """
        Forward pass of PSN.
        
        Args:
            x: Input tensor 
                - If obs_input_type="full": (batch_size, T_observation * N_agents * 4)
                - If obs_input_type="partial": (batch_size, T_observation * N_agents * 2)
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            mask: Binary mask of shape (batch_size, N_agents - 1)
        """
        # Determine input dimensions based on observation type
        if self.obs_input_type == "partial":
            input_dim = 2  # Only position (x, y)
        else:  # "full"
            input_dim = 4  # Full state (x, y, vx, vy)
        
        # Reshape input to (batch_size, T_observation, N_agents, input_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, T_observation, N_agents, input_dim)
        
        # Process each agent's trajectory through shared GRU
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
        
        # Concatenate all agent features: (batch_size, N_agents * gru_hidden_size)
        x = jnp.concatenate(agent_features, axis=1)
        
        # Apply MLP head for mask prediction
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f'mask_head_{i}')(x)
            x = nn.relu(x)
            if i < len(self.hidden_dims) - 1:  # Don't apply dropout to last layer
                x = nn.Dropout(rate=self.dropout_rate, name=f'mask_dropout_{i}')(x, deterministic=deterministic)
        
        # Mask output
        mask = nn.Dense(features=self.mask_output_dim, name='mask_output')(x)
        mask = nn.sigmoid(mask)  # Binary mask
        return mask

# ============================================================================
# ENHANCED TRAINING FUNCTIONS
# ============================================================================

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

def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray],
               batch_data: List[Dict[str, Any]], sigma1: float, sigma2: float, rng: jnp.ndarray = None, 
               obs_input_type: str = "full") -> Tuple[train_state.TrainState, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Single training step with game-solving loss function.
    
    Uses differentiable iLQR-based game solver for proper gradient flow.
    """
    observations, reference_trajectories = batch

    def loss_fn(params):
        predicted_masks = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        predicted_goals = extract_true_goals_from_batch(batch_data)
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        similarity_loss_val = batch_similarity_loss(predicted_masks, predicted_goals, batch_data, obs_input_type=obs_input_type)
        total_loss = similarity_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val
        return total_loss, (binary_loss_val, sparsity_loss_val, similarity_loss_val)
    
    (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

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
                   batch_size: int = 32, ego_agent_id: int = 0, obs_input_type: str = "full") -> Tuple[float, float, float, float, List[jnp.ndarray]]:
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
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(validation_data))
        batch_data = validation_data[start_idx:end_idx]
        
        # Prepare batch for validation
        observations, reference_trajectories = prepare_batch_for_training(batch_data, obs_input_type)
        
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
        similarity_loss_val = batch_similarity_loss(predicted_masks, predicted_goals, batch_data)
        
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

def train_psn(
        model: nn.Module,
        training_data: List[Dict[str, Any]],
        validation_data: List[Dict[str, Any]],
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        sigma1: float = 0.1,
        sigma2: float = 0.0,
        batch_size: int = 32,
        rng: jnp.ndarray = None,
        obs_input_type: str = "full"
) -> Tuple[List[float], List[float], List[float], List[float], List[float], 
           List[float], List[float], List[float], train_state.TrainState, str, float, int]:

    # setup log directories
    config_name = f"psn_gru_{obs_input_type}_planning_true_goals_N_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{learning_rate}_bs_{batch_size}_sigma1_{sigma1}_sigma2_{sigma2}_epochs_{num_epochs}"
    model_log_dir = os.path.join("log", config_name)
    os.makedirs(model_log_dir, exist_ok=True)
    print(f"This PSN type for training logs will be saved under: {model_log_dir}")

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
    input_shape = (batch_size, T_observation * N_agents * obs_dim)
    state = create_train_state(model, optimizer, input_shape, rng)

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

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_binary_losses = []
        epoch_sparsity_losses = []
        epoch_similarity_losses = []

        # Create batches
        num_batches = (len(training_data) + batch_size - 1) // batch_size

        batch_pbar = tqdm(range(num_batches), 
                         desc=f"Epoch {epoch+1}/{num_epochs} - Batches", 
                   position=1, leave=False)
        
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(training_data))
            batch_data = training_data[start_idx:end_idx]

            # Prepare batch for training with progress bar
            observations, reference_trajectories = prepare_batch_for_training(batch_data, obs_input_type)

            # Get predicted mask from PSN (validation mode, no dropout)
            rng, step_key = jax.random.split(rng)
            state, loss, (binary_loss_val, sparsity_loss_val, similarity_loss_val) = train_step(
                state,
                (observations, reference_trajectories),
                batch_data,
                sigma1,
                sigma2,
                step_key,
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
            state, validation_data, batch_size, ego_agent_id=ego_agent_id, obs_input_type=obs_input_type)
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
            best_model_path = os.path.join(log_dir, "psn_best_model.pkl")
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
    print("PSN Training with Pretrained Goal Inference Network")
    print("=" * 80)

    # Load reference trajectories
    reference_dir = os.path.join("src/data", config.training.data_dir)
    print(f"Loading reference trajectories from directory: {reference_dir}")
    training_data, validation_data = load_reference_trajectories(reference_dir)

    # create psn model with observation type
    psn_model = PlayerSelectionNetwork(
        hidden_dims=psn_hidden_dims,
        obs_input_type=config.psn.obs_input_type
    )

    # TODO: fix rng generation for now this is fixed to seed 42
    rng = jax.random.PRNGKey(config.training.seed)

    print(f"PSN model created with observation type: {config.psn.obs_input_type}")
    training_losses, validation_losses, binary_losses, sparsity_losses, \
    ego_agent_costs, validation_binary_losses, validation_sparsity_losses, \
    validation_ego_agent_costs, trained_state, log_dir, best_loss, best_epoch = train_psn(
        psn_model,
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








