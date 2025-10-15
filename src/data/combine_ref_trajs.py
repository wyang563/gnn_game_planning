#!/usr/bin/env python3
"""
Combine Reference Trajectories Script

This script combines individual ref_traj_sample_*.json files from a directory
into a single all_reference_trajectories.json file in the format expected by
the PSN training scripts.

Usage:
    python examples/combine_reference_trajectories.py [--input_dir INPUT_DIR] [--output_file OUTPUT_FILE]

Author: Assistant
Date: 2024
"""

import json
import os
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_individual_samples(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load all individual reference trajectory sample files from a directory.
    
    Args:
        input_dir: Directory containing ref_traj_sample_*.json files
        
    Returns:
        List of trajectory sample dictionaries
    """
    # Find all ref_traj_sample_*.json files in the directory
    pattern = os.path.join(input_dir, "ref_traj_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No ref_traj_sample_*.json files found in directory: {input_dir}")
    
    print(f"Found {len(json_files)} individual trajectory files in {input_dir}")
    
    # Load all samples
    reference_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
                reference_data.append(sample_data)
                print(f"  ✓ Loaded: {os.path.basename(json_file)}")
        except Exception as e:
            print(f"  ✗ Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(reference_data)} reference trajectory samples")
    return reference_data


def validate_sample_format(sample_data: Dict[str, Any], sample_id: int) -> bool:
    """
    Validate that a sample has the expected format.
    
    Args:
        sample_data: Sample data dictionary
        sample_id: Sample identifier for error reporting
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["sample_id", "init_positions", "target_positions", "trajectories", "metadata"]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in sample_data:
            print(f"  ✗ Sample {sample_id}: Missing required key '{key}'")
            return False
    
    # Check trajectories structure
    if not isinstance(sample_data["trajectories"], dict):
        print(f"  ✗ Sample {sample_id}: 'trajectories' should be a dictionary")
        return False
    
    # Check agent trajectories
    for agent_key, agent_data in sample_data["trajectories"].items():
        if not isinstance(agent_data, dict):
            print(f"  ✗ Sample {sample_id}: Agent {agent_key} data should be a dictionary")
            return False
        
        if "states" not in agent_data or "controls" not in agent_data:
            print(f"  ✗ Sample {sample_id}: Agent {agent_key} missing 'states' or 'controls'")
            return False
    
    return True


def combine_reference_trajectories(input_dir: str, output_file: str) -> None:
    """
    Combine individual reference trajectory files into a single JSON file.
    
    Args:
        input_dir: Directory containing individual ref_traj_sample_*.json files
        output_file: Path for the output all_reference_trajectories.json file
    """
    print("=" * 80)
    print("Combining Reference Trajectories")
    print("=" * 80)
    
    # Load individual samples
    reference_data = load_individual_samples(input_dir)
    
    if not reference_data:
        raise ValueError("No valid trajectory samples found")
    
    # Validate sample formats
    print(f"\nValidating sample formats...")
    valid_samples = []
    for i, sample_data in enumerate(reference_data):
        if validate_sample_format(sample_data, i):
            valid_samples.append(sample_data)
        else:
            print(f"  ✗ Skipping invalid sample {i}")
    
    if not valid_samples:
        raise ValueError("No valid trajectory samples found after validation")
    
    print(f"Valid samples: {len(valid_samples)}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write combined data to output file
    print(f"\nWriting combined data to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    
    print(f"✓ Successfully created {output_file}")
    print(f"  - Total samples: {len(valid_samples)}")
    print(f"  - File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Print sample structure info
    if valid_samples:
        sample = valid_samples[0]
        n_agents = len(sample["init_positions"])
        n_steps = len(sample["trajectories"]["agent_0"]["states"])
        print(f"  - Agents per sample: {n_agents}")
        print(f"  - Time steps per sample: {n_steps}")
        print(f"  - State dimension: {len(sample['trajectories']['agent_0']['states'][0])}")
        print(f"  - Control dimension: {len(sample['trajectories']['agent_0']['controls'][0])}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Combine individual reference trajectory files into a single JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine 4-player trajectories
  python examples/combine_reference_trajectories.py --input_dir reference_trajectories_4p --output_file reference_trajectories_4p/all_reference_trajectories.json
  
  # Combine 10-player trajectories  
  python examples/combine_reference_trajectories.py --input_dir reference_trajectories_10p --output_file reference_trajectories_10p/all_reference_trajectories.json
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='reference_trajectories_4p',
        help='Input directory containing ref_traj_sample_*.json files (default: reference_trajectories_4p)'
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='src/data/reference_trajectories_10p/all_reference_trajectories.json',
        help='Output file path for combined trajectories (default: reference_trajectories_4p/all_reference_trajectories.json)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    try:
        combine_reference_trajectories(args.input_dir, args.output_file)
        print("\n✓ Reference trajectory combination completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
