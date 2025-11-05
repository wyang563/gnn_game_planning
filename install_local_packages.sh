#!/bin/bash
# ==============================================================================
# Installation Script for Local Packages
# ==============================================================================
# This script installs all local packages (pip install -e .) as specified in
# the mmd/README.md file. Run this after creating and activating the conda
# environment and installing PyTorch.
#
# Usage:
#   conda activate gnn_game_planning
#   bash install_local_packages.sh

set -e  # Exit on error

echo "=================================================="
echo "Installing Local Packages for MMD"
echo "=================================================="

# Store the root directory
ROOT_DIR="$(pwd)"

# Check if we're in the gnn_game_planning directory
if [ ! -d "mmd/deps" ]; then
    echo "Error: This script must be run from the gnn_game_planning root directory"
    exit 1
fi

echo ""
echo "Step 1/4: Installing torch_robotics..."
cd mmd/deps/torch_robotics
pip install -e .
echo "✓ torch_robotics installed"

echo ""
echo "Step 2/4: Installing experiment_launcher..."
cd ../experiment_launcher
pip install -e .
echo "✓ experiment_launcher installed"

echo ""
echo "Step 3/4: Installing motion_planning_baselines..."
cd ../motion_planning_baselines
pip install -e .
echo "✓ motion_planning_baselines installed"

echo ""
echo "Step 4/4: Installing mmd package..."
cd ../../..  # Back to gnn_game_planning
cd mmd
pip install -e .
echo "✓ mmd package installed"

# Return to root directory
cd "$ROOT_DIR"

echo ""
echo "=================================================="
echo "✓ All local packages installed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run 'bash mmd/setup.sh' if needed (installs gdown)"
echo "  2. Download sample datasets and models (see mmd/README.md)"
echo "  3. Start using MMD and GNN game planning!"

