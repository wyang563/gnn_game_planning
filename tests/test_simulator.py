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

def test_simulator_basic():
    """Test basic simulator functionality."""
    print("Testing Simulator class...")
    
    # Create simulator with 5 agents in a 10x10 region
    sim = Simulator(num_agents=5, region_size=10.0)
    
    print(f"Initial states:")
    states = sim.get_states()
    print(f"Positions shape: {tuple(states['positions'].shape)}")
    print(f"Velocities shape: {tuple(states['velocities'].shape)}")
    print(f"Targets shape: {tuple(states['targets'].shape)}")
    
    # Run simulation for a few steps
    num_steps = 20
    trajectory_data = []
    
    for step in range(num_steps):
        # Get current states
        states = sim.get_states()
        trajectory_data.append(states)
        
        # Compute controls using Nash equilibrium (currently fallback)
        controls = sim.call()
        
        # Step the simulation
        sim.step(controls)
        
        if step % 5 == 0:
            print(f"Step {step}: Time = {states['time']:.2f}s")
            print(f"  Agent 0 position: [{states['positions'][0,0]:.2f}, {states['positions'][0,1]:.2f}]")
            print(f"  Agent 0 velocity: [{states['velocities'][0,0]:.2f}, {states['velocities'][0,1]:.2f}]")
    
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
        ax1.plot(float(positions[-1, 0]), float(positions[-1, 1]), 'ro', markersize=8)  # End
        
        # Mark target
        target = sim.targets[agent_id]
        ax1.plot(float(target[0]), float(target[1]), 'k*', markersize=12, alpha=0.5)
    
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
    plt.savefig('/home/alex/gnn_game_planning/simulator_test.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'simulator_test.png'")
    plt.show()

if __name__ == "__main__":
    try:
        # Test basic functionality
        print("=" * 50)
        print("PSN Game Simulator Test")
        print("=" * 50)
        
        # Create and test simulator
        sim = Simulator(num_agents=3, region_size=10.0)
        trajectory_data = test_simulator_basic()
        
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
