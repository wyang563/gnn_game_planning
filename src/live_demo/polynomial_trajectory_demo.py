"""
Simple demo to fly a single drone through a series of waypoints using polynomial trajectories.

This demonstrates smooth trajectory execution by generating polynomial trajectories and
streaming them to the drone at high frequency using goto commands.

Commands are executed sequentially using the same pattern as swarm_demo.py:
1. Arm
2. Takeoff
3. Execute trajectory (stream polynomial waypoints at 10Hz)
4. Land

Usage:
    python polynomial_trajectory_demo.py
    
Requirements:
    - One Crazyflie drone connected via radio
    - Update drone_uri in live_config.yaml to match your drone's address
"""

import sys
from pathlib import Path
import time
import numpy as np
import traceback
import threading
from queue import Queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from utils.trajectory_utils import simple_polynomial_trajectory
from utils.ops import Arm, Takeoff, Land, Goto, Quit, ExecuteTrajectory
from utils.config_parser import parse_config
from collections import namedtuple

# ===== LOAD CONFIGURATION =====
config = parse_config("src/live_demo/live_config.yaml")

# Extract configuration
STEP_TIME = config["step_time"]
LAND_TIME = config["land_time"]
XY_POSITION_RANGE = config["xy_position_range"]
Z_POSITION_RANGE = config["z_position_range"]

# Demo-specific configuration
DRONE_INDEX = 0  # Use first drone from config
DRONE_URI = f"radio://0/80/2M/E7E7E7E7{config['drone_uris'][DRONE_INDEX]}"
TAKEOFF_HEIGHT = 0.5  # meters
FLIGHT_DURATION = 10.0  # seconds

# Define waypoints to visit [x, y, z]
WAYPOINTS = np.array([
    [0.0, 0.0, TAKEOFF_HEIGHT],      # Start position
    [0.5, 0.0, TAKEOFF_HEIGHT],      # Move right
    [0.5, 0.5, TAKEOFF_HEIGHT + 0.2],  # Move forward and up
    [0.0, 0.5, TAKEOFF_HEIGHT],      # Move left
    [0.0, 0.0, TAKEOFF_HEIGHT],      # Return to start
])

# Define velocities at each waypoint [vx, vy, vz]
# Start and end with zero velocity for smooth takeoff/landing
VELOCITIES = np.array([
    [0.0, 0.0, 0.0],   # Stationary at start
    [0.3, 0.0, 0.0],   # Moving right
    [0.0, 0.3, 0.1],   # Moving forward and up
    [-0.3, 0.0, 0.0],  # Moving left
    [0.0, 0.0, 0.0],   # Stationary at end
])

# Command queue for drone control
controlQueue = Queue()

# Global to store trajectory pieces (needed for duration calculation)
TRAJECTORY_PIECES = None


def validate_waypoint_safety(position, xy_range, z_range):
    """
    Validate that a waypoint is safe (not NaN and within safety bounds).
    
    Args:
        position: Array-like [x, y, z] position
        xy_range: [min, max] allowed range for x and y coordinates
        z_range: [min, max] allowed range for z coordinate
        
    Raises:
        ValueError: If position contains NaN or is outside safety bounds
    """
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    
    # Check for NaN values
    if np.isnan(x) or np.isnan(y) or np.isnan(z):
        raise ValueError(f'SAFETY ERROR: Waypoint contains NaN values: ({x}, {y}, {z})')
    
    # Check x-y bounds
    if x < xy_range[0] - 0.5 or x > xy_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: x-coordinate {x:.3f} outside safety range [{xy_range[0]}, {xy_range[1]}]')
    if y < xy_range[0] - 0.5 or y > xy_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: y-coordinate {y:.3f} outside safety range [{xy_range[0]}, {xy_range[1]}]')
    
    # Check z bounds
    if z < 0.0 or z > z_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: z-coordinate {z:.3f} outside safety range [{z_range[0]}, {z_range[1]}]')
    
    return True


def arm_drone(scf):
    """Arm the drone motors"""
    print("Arming drone...")
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)
    print("✓ Drone armed")


def crazyflie_control(scf):
    """
    Control loop that processes commands from the queue.
    Uses command types from utils/ops.py for consistency.
    """
    cf = scf.cf
    commander = cf.high_level_commander
    
    # Set fade to color effect and reset to Led-ring OFF
    cf.param.set_value('ring.effect', '14')
    
    while True:
        command = controlQueue.get()
        
        if type(command) is Quit:
            return
        elif type(command) is Arm:
            print("  Executing: Arm")
            arm_drone(scf)
        elif type(command) is Takeoff:
            print(f"  Executing: Takeoff to {command.height:.2f}m")
            commander.takeoff(command.height, command.time)
        elif type(command) is Land:
            print(f"  Executing: Land")
            commander.land(0.0, command.time)
        elif type(command) is Goto:
            commander.go_to(command.x, command.y, command.z, 0, command.time)
        elif type(command) is ExecuteTrajectory:
            print(f"  Executing: Streaming polynomial trajectory")
            # Stream trajectory by evaluating polynomials at high frequency
            dt = 1.0 / command.sample_rate
            total_duration = sum(piece.duration for piece in command.pieces)
            
            print(f"  Sample rate: {command.sample_rate} Hz, Duration: {total_duration:.1f}s")
            
            t = 0.0
            piece_start_time = 0.0
            piece_idx = 0
            
            while t < total_duration and piece_idx < len(command.pieces):
                # Find which piece we're currently in
                current_piece = command.pieces[piece_idx]
                t_local = t - piece_start_time
                
                # Check if we need to move to next piece
                if t_local >= current_piece.duration and piece_idx < len(command.pieces) - 1:
                    piece_start_time += current_piece.duration
                    piece_idx += 1
                    current_piece = command.pieces[piece_idx]
                    t_local = t - piece_start_time
                
                # Evaluate position at this time
                x, y, z, _ = current_piece.evaluate(t_local)
                
                # Validate position
                try:
                    validate_waypoint_safety([x, y, z], XY_POSITION_RANGE, Z_POSITION_RANGE)
                except ValueError as e:
                    print(f"  ✗ Safety violation at t={t:.2f}s: {e}")
                    break
                
                # Send goto command
                commander.go_to(x, y, z, 0, dt * 1.5)
                
                # Progress indicator (every 0.5 seconds)
                if int(t * 2) != int((t - dt) * 2):
                    progress = (t / total_duration) * 100
                    print(f"    Progress: {progress:.0f}% - Position: ({x:.2f}, {y:.2f}, {z:.2f})")
                
                # Wait for next sample time
                time.sleep(dt)
                t += dt
            
            # Hold final position briefly
            final_piece = command.pieces[-1]
            x, y, z, yaw = final_piece.evaluate(final_piece.duration)
            commander.go_to(x, y, z, 0, 1.0)
            time.sleep(1.0)
            
            print(f"  ✓ Trajectory execution complete")
        else:
            print(f'Warning! Unknown command {command}')


def plot_polynomial_trajectory(waypoints, trajectory_pieces, sample_rate=50):
    """
    Plot the polynomial trajectory in 3D.
    
    Args:
        waypoints: Array of waypoints [x, y, z]
        trajectory_pieces: List of polynomial trajectory pieces
        sample_rate: Number of points to sample per second for smooth visualization
    """
    # Sample the trajectory at high frequency for smooth visualization
    sampled_positions = []
    
    for piece in trajectory_pieces:
        num_samples = int(piece.duration * sample_rate)
        t_samples = np.linspace(0, piece.duration, num_samples)
        
        for t in t_samples:
            x, y, z, yaw = piece.evaluate(t)
            sampled_positions.append([x, y, z])
    
    sampled_positions = np.array(sampled_positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the smooth trajectory
    ax.plot(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2],
            'b-', linewidth=2, label='Polynomial Trajectory', alpha=0.7)
    
    # Plot waypoints
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               c='red', s=100, marker='o', label='Waypoints', zorder=5)
    
    # Annotate waypoints
    for i, waypoint in enumerate(waypoints):
        ax.text(waypoint[0], waypoint[1], waypoint[2], f'  P{i}',
                fontsize=10, color='red')
    
    # Mark start and end points
    ax.scatter([waypoints[0, 0]], [waypoints[0, 1]], [waypoints[0, 2]],
               c='green', s=200, marker='*', label='Start', zorder=6)
    ax.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], [waypoints[-1, 2]],
               c='orange', s=200, marker='s', label='End', zorder=6)
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Polynomial Trajectory Path', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([
        sampled_positions[:, 0].max() - sampled_positions[:, 0].min(),
        sampled_positions[:, 1].max() - sampled_positions[:, 1].min(),
        sampled_positions[:, 2].max() - sampled_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (sampled_positions[:, 0].max() + sampled_positions[:, 0].min()) * 0.5
    mid_y = (sampled_positions[:, 1].max() + sampled_positions[:, 1].min()) * 0.5
    mid_z = (sampled_positions[:, 2].max() + sampled_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig('src/live_demo/plots/polynomial_trajectory.png')
    print("✓ Trajectory plot displayed")


def generate_trajectory():
    """
    Generate polynomial trajectory from waypoints.
    
    Returns:
        trajectory_pieces: List of polynomial trajectory pieces
    """
    print("\n=== GENERATING POLYNOMIAL TRAJECTORY ===")
    
    # Calculate durations for each segment
    num_segments = len(WAYPOINTS) - 1
    segment_duration = FLIGHT_DURATION / num_segments
    durations = np.full(num_segments, segment_duration)
    
    print(f"Waypoints: {len(WAYPOINTS)}")
    print(f"Segments: {num_segments}")
    print(f"Duration per segment: {segment_duration:.2f}s")
    
    # Validate all waypoints
    print("Validating waypoints...")
    for i, waypoint in enumerate(WAYPOINTS):
        try:
            validate_waypoint_safety(waypoint, XY_POSITION_RANGE, Z_POSITION_RANGE)
            print(f"  ✓ Waypoint {i}: ({waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f})")
        except ValueError as e:
            print(f"  ✗ Waypoint {i} failed validation: {e}")
            raise
    
    # Generate polynomial trajectory
    print("\nGenerating polynomial trajectory...")
    trajectory_pieces = simple_polynomial_trajectory(
        WAYPOINTS, 
        VELOCITIES, 
        durations, 
        yaw=0.0
    )
    print(f"✓ Generated {len(trajectory_pieces)} trajectory pieces")

    # Plot simple polynomial trajectory
    # plot_polynomial_trajectory(WAYPOINTS, trajectory_pieces)
    
    return trajectory_pieces

def control_thread(sequence):
    """
    Execute the sequence of commands step by step.
    Similar to swarm_demo.py but for single drone.
    
    Args:
        sequence: List of commands to execute
    """
    try:
        step_idx = 0
        
        while step_idx < len(sequence):
            command = sequence[step_idx]
            print(f'\nStep {step_idx}: {command}')
            
            # Send command to drone
            controlQueue.put(command)
            
            # Wait appropriate duration based on command type
            if type(command) is Arm:
                time.sleep(2.0)
            elif type(command) is Takeoff:
                time.sleep(command.time + 1.0)
            elif type(command) is Land:
                time.sleep(command.time + 1.0)
            elif type(command) is Goto:
                time.sleep(command.time + 0.5)
            elif type(command) is ExecuteTrajectory:
                # The trajectory execution happens inside the command handler
                # and blocks until complete, so no need to wait here
                pass
            
            step_idx += 1
        
        print("\n" + "=" * 60)
        print("SEQUENCE EXECUTION SUCCESSFUL!")
        print("=" * 60)

    except KeyboardInterrupt:
        print('\n!!! KEYBOARD INTERRUPT DETECTED !!!')
        print('!!! SAFE LANDING DRONE !!!')
        
        # Send land command to drone
        try:
            controlQueue.put(Land(LAND_TIME))
            print(f'   Landing command sent to drone')
        except Exception as land_error:
            print(f'   Failed to send land command to drone: {land_error}')
        
        # Wait for landing to complete
        print(f'   Waiting {LAND_TIME} seconds for safe landing...')
        time.sleep(LAND_TIME + 1.0)
        print('   Safe landing complete')
        
    except Exception as e:
        print(f'\n!!! ERROR DETECTED: {type(e).__name__}: {e}')
        print('\n!!! FULL STACK TRACE:')
        traceback.print_exc()
        print('\n!!! EMERGENCY LANDING DRONE !!!')
        
        # Send emergency land command to drone
        try:
            controlQueue.put(Land(LAND_TIME))
            print(f'   Emergency land command sent to drone')
        except Exception as land_error:
            print(f'   Failed to send land command to drone: {land_error}')
        
        # Wait for landing to complete
        print(f'   Waiting {LAND_TIME} seconds for emergency landing...')
        time.sleep(LAND_TIME + 1.0)
        print('   Emergency landing complete')
        
    finally:
        # Always send Quit command to cleanly exit drone control thread
        controlQueue.put(Quit())
        print('\nDemo complete!')


def main():
    """Main execution function"""
    global TRAJECTORY_PIECES
    
    print("=" * 70)
    print("POLYNOMIAL TRAJECTORY DEMO - Single Drone")
    print("=" * 70)
    print(f"Configuration loaded from: src/live_demo/live_config.yaml")
    print(f"Drone URI: {DRONE_URI}")
    print(f"Waypoints: {len(WAYPOINTS)}")
    print(f"Flight duration: {FLIGHT_DURATION}s")
    print(f"Safety bounds: XY=[{XY_POSITION_RANGE[0]}, {XY_POSITION_RANGE[1]}], "
          f"Z=[{Z_POSITION_RANGE[0]}, {Z_POSITION_RANGE[1]}]")
    print()
    
    # Generate trajectory before connecting to drone
    TRAJECTORY_PIECES = generate_trajectory()
    
    # Create command sequence
    sequence = [
        Arm(),
        Takeoff(TAKEOFF_HEIGHT, STEP_TIME),
        ExecuteTrajectory(TRAJECTORY_PIECES, sample_rate=10),  # Stream at 10 Hz
        Land(LAND_TIME)
    ]
    
    print(f"\nCreated sequence with {len(sequence)} commands")
    print("Sequence:")
    for i, cmd in enumerate(sequence):
        print(f"  {i}: {cmd}")
    
    # Initialize drivers
    cflib.crtp.init_drivers()
    
    print("\nConnecting to drone...")
    
    with SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("✓ Connected to drone")
        
        # Reset estimator
        print("Resetting position estimator...")
        scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        scf.cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)
        print("✓ Estimator reset")
        
        # Start control thread for drone (runs crazyflie_control in parallel)
        print("\nStarting drone control thread...")
        control_thread_obj = threading.Thread(target=crazyflie_control, args=(scf,))
        control_thread_obj.start()
        
        # Execute command sequence (control_thread handles all errors)
        print("Starting command execution...\n")
        control_thread(sequence)
        
        # Wait for control thread to finish
        control_thread_obj.join()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        time.sleep(1)


if __name__ == "__main__":
    main()
