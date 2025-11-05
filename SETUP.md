# Setup Guide for GNN Game Planning + MMD

This guide walks you through setting up a unified conda environment that supports both the GNN Game Planning project and the MMD (Multi-Robot Motion Planning with Diffusion Models) system.

## Prerequisites

- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html) or Anaconda
- CUDA-capable GPU (recommended for PyTorch with CUDA support)
- Linux (tested on Ubuntu 20.04/22.04)

## Installation Steps

### 1. Create and Activate the Conda Environment

From the `gnn_game_planning` root directory:

```bash
conda env create -f environment.yml
conda activate gnn_game_planning
```

This creates a minimal Python 3.10 environment.

### 2. Install All Dependencies (Recommended - Automated)

Use the automated installation script:

```bash
bash install_dependencies.sh
```

This script will:
- Install PyTorch with CUDA support (prompts for confirmation)
- Install all Python dependencies from requirements.txt
- Handle dependency conflicts intelligently
- Optionally install urdfpy (requires networkx 2.2)

**OR** Follow the manual installation steps below:

### 2-Manual: Install PyTorch with CUDA Support

Install PyTorch with CUDA support (adjust based on your CUDA version):

```bash
# For CUDA 11.8 (recommended for MMD)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Alternative CUDA versions:**
- For CUDA 12.1: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
- For CPU only: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

### 3-Manual: Install Python Dependencies

Install dependencies (excluding urdfpy to avoid conflicts):

```bash
grep -v "urdfpy>=" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt
```

**Note:** `urdfpy` requires `networkx==2.2`, which conflicts with newer packages. It's excluded by default. If you specifically need urdfpy, install it separately:
```bash
pip install urdfpy==0.0.22  # Will downgrade networkx to 2.2
```

Alternatively, use `urdf-parser-py` which is already installed.

### 4. Install Local Packages (MMD Dependencies)

Run the installation script to install MMD's local packages in editable mode:

```bash
bash install_local_packages.sh
```

This script installs:
- `torch_robotics` - Robotics utilities for PyTorch
- `experiment_launcher` - Experiment management
- `motion_planning_baselines` - Motion planning baseline algorithms
- `mmd` - The MMD package itself

### 5. Additional Setup for MMD

If you plan to use MMD features, run the setup script:

```bash
cd mmd
bash setup.sh
cd ..
```

### 6. Download Sample Datasets and Models (Optional)

If you want to run MMD examples, download the pre-trained models and datasets:

```bash
conda activate gnn_game_planning
cd mmd

# Download training data
gdown --id 1Onw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
tar -xJvf data_trajectories.tar.xz

# Download trained models (smaller version)
gdown --id 1idBod6n8u38skqMwe4PEeAUFR1TiMC8h
tar -xJvf data_trained_models_small.tar.xz
mv data_trained_models_small data_trained_models

cd ..
```

## Verification

To verify your installation:

```bash
conda activate gnn_game_planning
python -c "import torch; import torch_geometric; import jax; print('Core libraries OK')"
python -c "import mmd; import torch_robotics; print('MMD packages OK')"
```

## Quick Start Examples

### Running MMD Inference

```bash
conda activate gnn_game_planning
cd mmd/scripts/inference
python3 inference_multi_agent.py
```

### Using GNN Game Planning

```bash
conda activate gnn_game_planning
# Run your GNN game planning scripts here
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors, verify your CUDA version:
```bash
nvidia-smi
nvcc --version
```
Then install the matching PyTorch version.

### Dependency Conflicts (urdfpy/networkx)
**Problem:** `urdfpy 0.0.22` requires `networkx==2.2`, but other packages need newer versions.

**Solutions:**
1. **Recommended:** Skip urdfpy and use `urdf-parser-py` (already included)
2. Install urdfpy separately after other dependencies: `pip install urdfpy==0.0.22`
3. If you don't need URDF parsing, comment out both URDF packages

### Package Conflicts
If you encounter other package version conflicts:
```bash
pip install --upgrade pip
# Install without problematic packages
grep -v "urdfpy" requirements.txt | grep -v "package_name" > temp_requirements.txt
pip install -r temp_requirements.txt
```

### Import Errors
If you get import errors for local packages:
```bash
bash install_local_packages.sh
```

### zarr Version Error
**Problem:** `zarr>=3.0.0` requires Python 3.11+

**Solution:** Already fixed in requirements.txt (uses `zarr>=2.10.0,<3.0.0`)

## Environment Details

- **Python Version:** 3.10
- **Primary Framework:** PyTorch 2.0.0
- **Additional Frameworks:** JAX, TensorFlow
- **CUDA Support:** Yes (via conda installation)

## Directory Structure

```
gnn_game_planning/
├── environment.yml              # Conda environment specification
├── requirements.txt             # Unified pip requirements
├── install_local_packages.sh    # Script to install local packages
├── SETUP.md                     # This file
├── mmd/                         # MMD subproject
│   ├── deps/                    # Local dependencies
│   │   ├── torch_robotics/
│   │   ├── experiment_launcher/
│   │   └── motion_planning_baselines/
│   ├── scripts/                 # Example scripts
│   └── ...
└── src/                         # GNN game planning source
```

## Additional Resources

- [MMD Paper](https://arxiv.org/abs/2410.03072)
- [MMD Project Page](https://multi-robot-diffusion.github.io)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

## Notes

- The unified `requirements.txt` resolves version conflicts between GNN Game Planning and MMD dependencies
- Some packages have specific version constraints (e.g., `setuptools~=70.2.0`, `einops~=0.6.1`)
- For development, consider installing additional tools: `pytest`, `black`, `flake8`
- The environment uses Python 3.10 for compatibility with both projects

