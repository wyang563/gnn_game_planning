#!/usr/bin/env python3
"""
Script to add z-coordinates to 2D trajectory CSV files.

For each agent, assigns a start and end height, then linearly interpolates
z-coordinates across all timesteps.

Usage:
    python add_z_coordinates.py <input_csv_file> [--z-min Z_MIN] [--z-max Z_MAX] [--output OUTPUT_FILE]

Example:
    python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv
    python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv --z-min 0.5 --z-max 2.5
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def add_z_coordinates(
    input_file: str,
    output_file: str = None,
    z_min: float = 0.0,
    z_max: float = 1.5,
    randomize_heights: bool = True,
    seed: int = 42,
    scale_xy: float = 1.0
) -> None:
    """
    Add z-coordinates to a 2D trajectory CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: same directory as input with '3D_trajectories.csv' name)
        z_min: Minimum height for agents
        z_max: Maximum height for agents
        randomize_heights: If True, randomly assign start/end heights per agent within [z_min, z_max]
        seed: Random seed for reproducibility
        scale_xy: Scale factor to multiply all x and y coordinates by (default: 1.0)
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get unique agent IDs
    agent_ids = sorted(df['agent_id'].unique())
    n_agents = len(agent_ids)
    
    print(f"Processing {n_agents} agents from {input_file}")
    
    # Initialize z column
    df['z'] = 0.0
    
    # For each agent, assign start/end heights and interpolate
    for agent_id in agent_ids:
        # Get all rows for this agent
        agent_mask = df['agent_id'] == agent_id
        agent_df = df[agent_mask].copy()
        
        # Get the timesteps for this agent (sorted)
        timesteps = agent_df['timestep'].values
        min_timestep = timesteps.min()
        max_timestep = timesteps.max()
        n_timesteps = len(timesteps)
        
        # Assign start and end heights for this agent
        if randomize_heights:
            z_start = np.random.uniform(z_min, z_max)
            z_end = np.random.uniform(z_min, z_max)
        else:
            # Evenly distribute agents in height space
            agent_idx = agent_ids.index(agent_id)
            height_range = z_max - z_min
            z_start = z_min + (agent_idx / max(1, n_agents - 1)) * height_range if n_agents > 1 else (z_min + z_max) / 2
            z_end = z_start  # Keep same height throughout
        
        # Create linearly interpolated z values from start to end
        # We interpolate based on the order of timesteps, not their absolute values
        z_values = np.linspace(z_start, z_end, n_timesteps)
        
        # Assign z values to the dataframe
        df.loc[agent_mask, 'z'] = z_values
        
        print(f"  Agent {agent_id}: z_start={z_start:.3f}, z_end={z_end:.3f}, timesteps={min_timestep} to {max_timestep}")
    
    # Scale x and y coordinates if scale_xy is not 1.0
    if scale_xy != 1.0:
        if 'x' in df.columns:
            df['x'] = df['x'] * scale_xy
        if 'y' in df.columns:
            df['y'] = df['y'] * scale_xy
        print(f"\nScaled x and y coordinates by factor: {scale_xy}")
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / "3D_trajectories.csv"
    
    # Reorder columns to put z after y
    column_order = []
    for col in df.columns:
        column_order.append(col)
        if col == 'y':
            column_order.append('z')
    
    # Remove duplicate 'z' if it was already in the list
    column_order = [col for i, col in enumerate(column_order) if col != 'z' or column_order[:i].count('z') == 0]
    
    # Make sure z is in the right place (after y, before vx)
    cols = list(df.columns)
    if 'z' in cols:
        cols.remove('z')
    
    # Find position to insert z (after y)
    y_idx = cols.index('y')
    cols.insert(y_idx + 1, 'z')
    
    df = df[cols]
    
    # Save to output file
    df.to_csv(output_file, index=False)
    print(f"\nSaved 3D trajectories to: {output_file}")
    print(f"Output shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Add z-coordinates to 2D trajectory CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (creates 3D_trajectories.csv in same directory)
  python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv
  
  # Specify height range
  python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv --z-min 0.5 --z-max 2.5
  
  # Specify output file
  python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv --output my_output.csv
  
  # Use deterministic heights (agents evenly distributed in height, no change over time)
  python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv --no-randomize
  
  # Scale xy coordinates by 1.5
  python add_z_coordinates.py plots/diffusion_trajs/four_agent_swarm/trajectories.csv --scale-xy 1.5
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file containing 2D trajectories'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: 3D_trajectories.csv in same directory as input)'
    )
    
    parser.add_argument(
        '--z-min',
        type=float,
        default=0.5,
        help='Minimum height for agents (default: 0.0)'
    )
    
    parser.add_argument(
        '--z-max',
        type=float,
        default=1.5,
        help='Maximum height for agents (default: 2.0)'
    )
    
    parser.add_argument(
        '--no-randomize',
        action='store_true',
        help='Use deterministic heights instead of random (agents evenly distributed in height space)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--scale-xy',
        type=float,
        default=1.0,
        help='Scale factor to multiply all x and y coordinates by (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Run the conversion
    add_z_coordinates(
        input_file=args.input_file,
        output_file=args.output,
        z_min=args.z_min,
        z_max=args.z_max,
        randomize_heights=not args.no_randomize,
        seed=args.seed,
        scale_xy=args.scale_xy
    )
    
    return 0

if __name__ == "__main__":
    exit(main())

