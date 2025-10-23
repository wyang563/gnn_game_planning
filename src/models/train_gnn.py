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
class GNNSelectionNetwork(nn.Module):
    """
    GNN Selection Network for selecting important agents.
    """
    hidden_dims: List[int]
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    mask_output_dim: int = N_agents - 1
    obs_input_type: str = "full"     # "full" or "partial"
    coarse_topk: int = 8             # pruning: keep K per destination
    gate_hidden: int = 32
    msg_hidden: int = 64
    upd_hidden: int = 64
    infl_hidden: int = 64
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
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
        
        # stack agent features 
        h_nodes = jnp.stack(agent_features, axis=1) # (B, N, gru_hidden_dim)

        # construct graph
        graph_ij = jnp.zeros((batch_size, N_agents, N_agents, 2 * self.gru_hidden_size))


        


