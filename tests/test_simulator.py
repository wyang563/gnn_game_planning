#!/usr/bin/env python3
"""
Test script for the PSN Game Simulator implementation.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from game.sim import Simulator

def _tensor_to_list(t):
    if isinstance(t, (int, float)):
        return t
    if hasattr(t, 'detach'):
        return t.detach().cpu().tolist()
    return t

def save_trajectories_json(trajectory_data, sim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    serializable = []
    for entry in trajectory_data:
        serializable.append({
            'positions': _tensor_to_list(entry['positions']),
            'velocities': _tensor_to_list(entry['velocities']),
            'targets': _tensor_to_list(entry['targets']),
            'time': float(entry['time']),
        })
    out_path = os.path.join(out_dir, 'simulator_trajectories.json')
    import json
    with open(out_path, 'w') as f:
        json.dump(serializable, f)
    print(f"Saved trajectories to '{out_path}'")

def test_simulator_basic(sim, num_steps=50):
    """Test basic simulator functionality."""
    print("Testing Simulator class...")
    print(f"Initial states:")
    states = sim.get_states()
    print(f"Positions shape: {tuple(states['positions'].shape)}")
    print(f"Velocities shape: {tuple(states['velocities'].shape)}")
    print(f"Targets shape: {tuple(states['targets'].shape)}")
    
    # Run simulation for a few steps
    trajectory_data = []
    
    for _ in range(num_steps):
        # Get current states
        states = sim.get_states()
        trajectory_data.append(states)
        
        # Compute controls using Nash equilibrium (currently fallback)
        controls = sim.call()
        
        # Step the simulation
        sim.step(controls)
        
        # if step % 5 == 0:
        #     print(f"Step {step}: Time = {states['time']:.2f}s")
        #     print(f"  Agent 0 position: [{states['positions'][0,0]:.2f}, {states['positions'][0,1]:.2f}]")
        #     print(f"  Agent 0 velocity: [{states['velocities'][0,0]:.2f}, {states['velocities'][0,1]:.2f}]")
    
    return trajectory_data

def visualize_trajectories(trajectory_data, sim):
    """Visualize agent trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trajectories
    for agent_id in range(sim.num_agents):
        positions = torch.stack([traj['positions'][agent_id] for traj in trajectory_data], dim=0)
        ax1.plot(positions[:, 0].cpu().numpy(), positions[:, 1].cpu().numpy(), 'o-', alpha=0.7, label=f'Agent {agent_id}')
        
        # Mark start and end
        ax1.plot(float(positions[0, 0]), float(positions[0, 1]), 'go', markersize=8)  # Start
        ax1.text(float(positions[0, 0]) + 0.05, float(positions[0, 1]) + 0.05, f"{agent_id}", color='g', fontsize=9, ha='left', va='bottom')
        ax1.plot(float(positions[-1, 0]), float(positions[-1, 1]), 'ro', markersize=8)  # End
        ax1.text(float(positions[-1, 0]) + 0.05, float(positions[-1, 1]) + 0.05, f"{agent_id}", color='r', fontsize=9, ha='left', va='bottom')
        
        # Mark target
        target = sim.targets[agent_id]
        ax1.plot(float(target[0]), float(target[1]), 'k*', markersize=12, alpha=0.5)
        ax1.text(float(target[0]) + 0.05, float(target[1]) + 0.05, f"{agent_id}", color='k', fontsize=9, ha='left', va='bottom')
    
    ax1.set_xlim(0, sim.region_size)
    ax1.set_ylim(0, sim.region_size)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Agent Trajectories')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot velocity magnitudes over time
    for agent_id in range(sim.num_agents):
        velocities = torch.stack([traj['velocities'][agent_id] for traj in trajectory_data], dim=0)
        vel_magnitudes = torch.norm(velocities, dim=1)
        times = torch.tensor([traj['time'] for traj in trajectory_data], dtype=torch.float32)
        ax2.plot(times.cpu().numpy(), vel_magnitudes.cpu().numpy(), 'o-', alpha=0.7, label=f'Agent {agent_id}')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity Magnitude (m/s)')
    ax2.set_title('Velocity Magnitudes Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, 'simulator_test.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{img_path}'")
    plt.show()

if __name__ == "__main__":
    try:
        # Test basic functionality
        print("=" * 50)
        print("PSN Game Simulator Test")
        print("=" * 50)
        
        # Create and test simulator
        solver_params = {
            'Q_goal': 5.0,
            'Q_prox': 10.0,
            'R': 0.1,
            'safety_radius': 0.3
        }
        sim = Simulator(num_agents=5, region_size=10.0, solver_params=solver_params)
        trajectory_data = test_simulator_basic(sim, num_steps=100)
        # Save trajectories as JSON in tests/outputs
        save_trajectories_json(trajectory_data, sim, os.path.join(os.path.dirname(__file__), 'outputs'))
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        print("=" * 50)
        
        # Create visualization
        try:
            visualize_trajectories(trajectory_data, sim)
        except ImportError:
            print("Matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"Visualization failed: {e}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
