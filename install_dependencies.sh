#!/bin/bash
# ==============================================================================
# Dependency Installation Script
# ==============================================================================
# This script installs all dependencies in a specific order to avoid conflicts
#
# Usage:
#   conda activate gnn_game_planning
#   bash install_dependencies.sh

set -e  # Exit on error

echo "=================================================="
echo "Installing Dependencies for GNN Game Planning + MMD"
echo "=================================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "gnn_game_planning" ]]; then
    echo "Error: Please activate the gnn_game_planning environment first:"
    echo "  conda activate gnn_game_planning"
    exit 1
fi

echo ""
echo "Step 1: Installing PyTorch with CUDA support..."
echo "Note: Adjust CUDA version based on your system (nvidia-smi to check)"
read -p "Install PyTorch with CUDA 11.8? (y/n, default: y): " install_pytorch
install_pytorch=${install_pytorch:-y}

if [[ "$install_pytorch" == "y" || "$install_pytorch" == "Y" ]]; then
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    echo "✓ PyTorch installed"
else
    echo "Skipping PyTorch installation. Make sure to install it manually!"
fi

echo ""
echo "Step 2: Installing core dependencies (without urdfpy)..."
# Create a temporary requirements file without urdfpy to avoid networkx conflicts
grep -v "urdfpy" /home/alex/gnn_game_planning/requirements.txt > /tmp/requirements_temp.txt || true
pip install -r /tmp/requirements_temp.txt
echo "✓ Core dependencies installed"

echo ""
echo "Step 3: Handling urdfpy (requires old networkx)..."
read -p "Install urdfpy 0.0.22? This will downgrade networkx to 2.2 (y/n, default: n): " install_urdfpy
install_urdfpy=${install_urdfpy:-n}

if [[ "$install_urdfpy" == "y" || "$install_urdfpy" == "Y" ]]; then
    pip install urdfpy==0.0.22
    echo "✓ urdfpy installed (networkx downgraded to 2.2)"
    echo "  Note: urdf-parser-py is available as an alternative"
else
    echo "Skipping urdfpy. Using urdf-parser-py as alternative."
    echo "  If you need urdfpy later, install it manually:"
    echo "  pip install urdfpy==0.0.22"
fi

echo ""
echo "=================================================="
echo "✓ Dependency installation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run 'bash install_local_packages.sh' to install MMD packages"
echo "  2. Run 'bash mmd/setup.sh' if needed"
echo "  3. Start using the environment!"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; import torch_geometric; import jax; print(\"✓ Core libraries OK\")'"

