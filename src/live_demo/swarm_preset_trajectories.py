import sys
from pathlib import Path

# Add parent directory to path to allow importing from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
import traceback
from queue import Queue
import random
import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from utils.ops import Arm, Takeoff, Land, Quit, ExecuteTrajectory
from utils.config_parser import parse_config
from utils.trajectory_utils import simple_polynomial_trajectory
from load_config import load_config
from utils.plot import (
    plot_drone_agent_trajs,
    plot_drone_agent_trajs_gif,
)

config = parse_config("src/live_demo/live_config.yaml")
main_config = load_config()

# setup drone radio URIs
uris = []
for i, uri_num in enumerate(config["drone_uris"]):
    interface_num = i // 3
    uris.append(f"radio://{interface_num}/80/2M/E7E7E7E7{uri_num}")

# Global brightness scale for drone ring colors (0 < scale <= 1)
DRONE_COLOR_BRIGHTNESS_SCALE = 0.3

# Predefined list of 10 distinct RGB colors (full intensity 0–255) for the drones
DRONE_COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 128, 255),    # light blue
    (0, 255, 128),    # spring green
]

# DEFAULT_SPEED = config["default_speed"]
STEP_TIME = config["step_time"]
LAND_TIME = config["land_time"]
MIN_GOAL_DISTANCE = config["min_goal_separation_distance"]
XY_POSITION_RANGE = config["xy_position_range"]
Z_POSITION_RANGE = config["z_position_range"]
STAGGER = config["stagger"]
T_total = config["T_total"]
PAUSE_AT_START = config.get("pause_at_start_duration", 2.0)  # seconds to pause at starting position

# Game theory optimization parameters
dt = main_config.game.dt
planning_horizon = config["T_total"]
mask_horizon = main_config.game.T_observation
opt_config = main_config.optimization.drone
num_iters = opt_config.num_iters
step_size = opt_config.step_size
collision_weight = opt_config.collision_weight
collision_scale = opt_config.collision_scale
control_weight = opt_config.control_weight
Q = jnp.diag(jnp.array(opt_config.Q))
R = jnp.diag(jnp.array(opt_config.R))
x_dim = opt_config.state_dim
pos_dim = x_dim // 2
u_dim = opt_config.control_dim
mask_threshold = config["mask_threshold"]

def generate_straight_line_reference_trajectory(start_pos, goal_pos, num_steps):
    """
    Generate a straight-line reference trajectory from start to goal.
    
    Args:
        start_pos: Starting 3D position [x, y, z]
        goal_pos: Goal 3D position [x, y, z]
        num_steps: Number of timesteps in the trajectory
    
    Returns:
        numpy array of shape (num_steps, 3) containing positions at each timestep
    """
    start = jnp.array(start_pos)
    goal = jnp.array(goal_pos)
    
    # Create linear interpolation from start to goal
    trajectory = jnp.linspace(start, goal, num_steps)
    
    return trajectory

def activate_mellinger_controller(scf, use_mellinger):
    controller = 1
    if use_mellinger:
        controller = 2
    scf.cf.param.set_value('stabilizer.controller', str(controller))

def arm(scf):
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)

def crazyflie_control(scf):
    cf = scf.cf
    drone_index = uris.index(cf.link_uri)
    control = controlQueues[drone_index]

    activate_mellinger_controller(scf, False)

    commander = scf.cf.high_level_commander

    # Assign a unique, dimmed LED ring color to this drone based on its index.
    # Uses the standard LED-ring parameters (fadeColor + effect) when available.
    try:
        base_r, base_g, base_b = DRONE_COLORS[drone_index % len(DRONE_COLORS)]
        # Apply global brightness scaling to reduce intensity
        r = int(base_r * DRONE_COLOR_BRIGHTNESS_SCALE)
        g = int(base_g * DRONE_COLOR_BRIGHTNESS_SCALE)
        b = int(base_b * DRONE_COLOR_BRIGHTNESS_SCALE)
        # Pack RGB into a 24-bit integer 0xRRGGBB
        color_value = (int(r) << 16) | (int(g) << 8) | int(b)
        cf.param.set_value('ring.fadeColor', str(color_value))
        # Effect 14 is typically "fade to color" using fadeColor
        cf.param.set_value('ring.effect', '14')
    except KeyError as e:
        print(f'Warning: could not set LED ring color for drone {drone_index}: {e}')
    except Exception as e:
        print(f'Warning: unexpected error while setting LED color for drone {drone_index}: {e}')

    while True:
        command = control.get()
        if type(command) is Quit:
            return
        elif type(command) is Arm:
            arm(scf)
        elif type(command) is Takeoff:
            commander.takeoff(command.height, command.time)
        elif type(command) is Land:
            commander.land(0.0, command.time)
            # After landing, turn off the LED ring effect
            try:
                # Wait approximately for the landing duration to complete
                time.sleep(command.time)
                cf.param.set_value('ring.effect', '0')
            except KeyError as e:
                print(f'Warning: could not turn off LED ring for drone {drone_index}: {e}')
            except Exception as e:
                print(f'Warning: unexpected error while turning off LED ring for drone {drone_index}: {e}')
        elif type(command) is ExecuteTrajectory:
            # Stream trajectory by evaluating polynomials at high frequency
            dt = 1.0 / command.sample_rate
            total_duration = sum(piece.duration for piece in command.pieces)
            
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
                    validate_waypoint_safety([x, y, z], uris.index(cf.link_uri), XY_POSITION_RANGE, Z_POSITION_RANGE)
                except ValueError as e:
                    print(f"  ✗ Safety violation at t={t:.2f}s: {e}")
                    break
                
                # Send goto command
                commander.go_to(x, y, z, 0, dt * 1.5)
                
                # Wait for next sample time
                time.sleep(dt)
                t += dt
            
            # Hold final position briefly
            final_piece = command.pieces[-1]
            x, y, z, _ = final_piece.evaluate(final_piece.duration)
            commander.go_to(x, y, z, 0, 1.0)
            time.sleep(1.0)
        else:
            print('Warning! unknown command {} for uri {}'.format(command, cf.uri))

def validate_waypoint_safety(position, agent_id, xy_range, z_range):
    """
    Validate that a waypoint is safe (not NaN and within safety bounds).
    
    Args:
        position: Array-like [x, y, z] position
        agent_id: ID of the agent for error messages
        xy_range: [min, max] allowed range for x and y coordinates
        z_range: [min, max] allowed range for z coordinate
        
    Raises:
        ValueError: If position contains NaN or is outside safety bounds
    """
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    
    # Check for NaN values
    if jnp.isnan(x) or jnp.isnan(y) or jnp.isnan(z):
        raise ValueError(f'SAFETY ERROR: Drone {agent_id} waypoint contains NaN values: ({x}, {y}, {z})')
    
    # Check x-y bounds
    if x < xy_range[0] - 0.5 or x > xy_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: Drone {agent_id} x-coordinate {x:.3f} outside safety range [{xy_range[0]}, {xy_range[1]}]')
    if y < xy_range[0] - 0.5 or y > xy_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: Drone {agent_id} y-coordinate {y:.3f} outside safety range [{xy_range[0]}, {xy_range[1]}]')
    
    # Check z bounds
    if z < 0.0 or z > z_range[1] + 0.5:
        raise ValueError(f'SAFETY ERROR: Drone {agent_id} z-coordinate {z:.3f} outside safety range [{z_range[0]}, {z_range[1]}]')
    
    return True

def load_fixed_trajectories(trajectory_file, num_agents):
    """
    Load pre-computed trajectories from a single CSV file.
    
    Args:
        trajectory_file: Path to CSV file containing trajectories in format:
                        agent_id,timestep,x,y,z,vx,vy,is_start,is_goal
                        Can also be in the old format with separate files per agent.
        num_agents: Number of agents
        
    Returns:
        x_trajs: Array of shape (n_agents, T_total, x_dim) containing positions and velocities
    """
    from pathlib import Path
    
    trajectory_path = Path(trajectory_file)
    
    # Check if it's a directory (old format) or a file (new format)
    if trajectory_path.is_dir():
        # Old format: separate files per agent
        print(f'Loading trajectories from directory (old format): {trajectory_path}')
        trajectories = []
        
        for agent_id in range(num_agents):
            traj_file = trajectory_path / f'trajectory_agent{agent_id}.csv'
            if not traj_file.exists():
                raise FileNotFoundError(f'Trajectory file not found: {traj_file}')
            
            # Load trajectory data: px, py, pz, vx, vy, vz
            traj_data = np.loadtxt(traj_file, delimiter=',', skiprows=1)
            trajectories.append(traj_data)
        
        x_trajs = np.array(trajectories)  # (n_agents, T_total, x_dim)
    
    else:
        # New format: single CSV file with all agents
        print(f'Loading trajectories from single CSV file: {trajectory_path}')
        
        if not trajectory_path.exists():
            raise FileNotFoundError(f'Trajectory file not found: {trajectory_path}')
        
        # Load CSV file
        df = pd.read_csv(trajectory_path)
        
        # Check required columns
        required_cols = ['agent_id', 'timestep', 'x', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f'CSV must contain columns: {required_cols}')
        
        # Check if z column exists
        has_z = 'z' in df.columns
        has_vx = 'vx' in df.columns
        has_vy = 'vy' in df.columns
        
        # Filter out rows with timestep == -1 (initial positions) and goal positions
        df_filtered = df[df['timestep'] >= 0].copy()
        
        # Sort by agent_id and timestep
        df_filtered = df_filtered.sort_values(['agent_id', 'timestep'])
        
        trajectories = []
        agent_ids = sorted(df_filtered['agent_id'].unique())
        
        if len(agent_ids) != num_agents:
            print(f'Warning: Expected {num_agents} agents, but found {len(agent_ids)} in CSV')
            print(f'Found agent IDs: {agent_ids}')
        
        for agent_id in agent_ids[:num_agents]:
            agent_data = df_filtered[df_filtered['agent_id'] == agent_id].copy()
            
            # Extract positions
            px = agent_data['x'].values
            py = agent_data['y'].values
            pz = agent_data['z'].values if has_z else np.zeros_like(px)
            
            # Extract or compute velocities
            if has_vx and has_vy:
                vx = agent_data['vx'].values
                vy = agent_data['vy'].values
            else:
                # Compute velocities from positions using finite differences
                vx = np.gradient(px)
                vy = np.gradient(py)
            
            # Compute vz from z positions
            vz = np.gradient(pz)
            
            # Combine into trajectory array: [px, py, pz, vx, vy, vz]
            traj = np.stack([px, py, pz, vx, vy, vz], axis=1)
            trajectories.append(traj)
        
        x_trajs = np.array(trajectories)  # (n_agents, T_total, x_dim)
    
    print(f'  Shape: {x_trajs.shape}')
    print(f'  Number of agents: {x_trajs.shape[0]}')
    print(f'  Number of timesteps: {x_trajs.shape[1]}')
    
    return jnp.array(x_trajs)


def control_thread_fixed_trajectory(swarm, fixed_x_trajs, goals, opt_params):
    """
    Execute a fixed pre-computed trajectory in one go.
    
    Args:
        swarm: Swarm object to query drone positions
        fixed_x_trajs: Pre-computed trajectories (n_agents, T_total, x_dim)
        goals: List of 3D goal positions [x, y, z] for each agent
        opt_params: Dictionary of optimization parameters
    """
    step_idx = 0
    num_agents = fixed_x_trajs.shape[0]
    T_total_traj = fixed_x_trajs.shape[1]
    
    # Extract parameters
    pos_dim = opt_params['pos_dim']
    dt = opt_params['dt']
    
    # Store the executed trajectory for plotting
    init_states = None

    try:
        while True:
            if step_idx >= len(sequence):
                print('Reaching the end of the sequence, stopping!')
                break
            
            step_commands = sequence[step_idx]
            print(f'\nStep {step_idx}:')
            
            # Send all commands in this step to their respective drones
            max_duration = 0
            
            for cf_id, command in step_commands:
                print(f' - Running: {command} on drone {cf_id}')
                controlQueues[cf_id].put(command)
                
                # Track the longest duration in this step
                if type(command) is ExecuteTrajectory:
                    # Calculate total duration for the trajectory
                    traj_duration = sum(piece.duration for piece in command.pieces)
                    max_duration = max(max_duration, traj_duration)
                elif type(command) is Takeoff:
                    max_duration = max(max_duration, command.time)
                elif type(command) is Land:
                    max_duration = max(max_duration, command.time)
                elif type(command) is Arm:
                    max_duration = max(max_duration, 1.5)
            
            # Wait for all commands in this step to complete
            if max_duration > 0:
                time.sleep(max_duration + 0.5)
            
            # Handle post-command actions based on command type
            if step_commands and type(step_commands[0][1]) is Takeoff:
                print('   Takeoff complete for all drones')
                
                # Get absolute positions of each drone directly from swarm
                positions_dict = swarm.get_estimated_positions()
                init_positions_list = []
                for uri in uris:
                    pos = positions_dict.get(uri)
                    if pos:
                        position = [pos.x, pos.y, pos.z]
                    else:
                        raise ValueError(f'Position not available for drone {uri}')
                    init_positions_list.append(position)
                    print(f'   Drone {uris.index(uri)} at position after takeoff: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]')
                
                # Initialize states with positions and zero velocities: [x, y, z, vx, vy, vz]
                init_states = jnp.array([[pos[0], pos[1], pos[2], 0.0, 0.0, 0.0] for pos in init_positions_list])
                
                print('\n   Generating trajectories to starting positions...')
                
                # Generate trajectories to fly each drone to its first trajectory waypoint
                fly_to_start_commands = []
                for agent_id in range(num_agents):
                    current_pos = init_positions_list[agent_id]
                    start_pos = fixed_x_trajs[agent_id, 0, :pos_dim]  # First waypoint
                    
                    print(f'   Agent {agent_id}: Flying from [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] '
                          f'to start [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]')
                    
                    # Validate start position
                    validate_waypoint_safety(start_pos, agent_id, XY_POSITION_RANGE, Z_POSITION_RANGE)
                    
                    # Create a simple 2-waypoint trajectory: current -> start
                    waypoints_np = np.array([current_pos, start_pos])
                    velocities_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Zero velocity at both ends
                    durations_np = np.array([2.0])  # 2 seconds to fly to start
                    
                    trajectory_pieces = simple_polynomial_trajectory(
                        waypoints_np,
                        velocities_np,
                        durations_np,
                        yaw=0.0
                    )
                    
                    fly_to_start_commands.append((agent_id, ExecuteTrajectory(trajectory_pieces, sample_rate=10)))
                
                # Update sequence with fly-to-start commands
                sequence[2] = fly_to_start_commands.copy()
                print('   ✓ Fly-to-start trajectory generation complete')

            elif step_commands and type(step_commands[0][1]) is ExecuteTrajectory:
                # Check if this was the fly-to-start step (sequence[2]) or the full trajectory (sequence[3])
                if step_idx == 2:
                    print('   ✓ Drones arrived at starting positions')
                    print(f'   Pausing for {PAUSE_AT_START} seconds...')
                    time.sleep(PAUSE_AT_START)
                    print('   ✓ Pause complete, generating full trajectory...')
                    
                    # Now generate the full trajectory from the fixed waypoints
                    full_trajectory_commands = []
                    for agent_id in range(num_agents):
                        # Extract all waypoints and velocities from fixed trajectory
                        waypoints = []
                        velocities = []
                        
                        # Use all timesteps from the fixed trajectory
                        for t in range(T_total_traj):
                            state = fixed_x_trajs[agent_id, t, :]
                            waypoints.append(state[:pos_dim])
                            velocities.append(state[pos_dim:])
                        
                        waypoints = jnp.array(waypoints)
                        velocities = jnp.array(velocities)
                        
                        # Validate all waypoints before generating trajectory
                        for i, waypoint in enumerate(waypoints):
                            try:
                                validate_waypoint_safety(waypoint, agent_id, XY_POSITION_RANGE, Z_POSITION_RANGE)
                            except ValueError as e:
                                print(f'  ✗ Waypoint {i} for agent {agent_id} failed validation: {e}')
                                raise
                        
                        # Calculate duration for each segment (time between waypoints)
                        # We have T_total_traj waypoints, so T_total_traj-1 segments
                        durations = jnp.full(T_total_traj - 1, dt)
                        
                        # Convert to numpy arrays for trajectory generation
                        waypoints_np = np.array(waypoints)
                        velocities_np = np.array(velocities)
                        durations_np = np.array(durations)
                        
                        print(f'   Agent {agent_id}: Generating trajectory with {len(waypoints_np)} waypoints, total duration: {durations_np.sum():.2f}s')
                        
                        # Generate polynomial trajectory
                        trajectory_pieces = simple_polynomial_trajectory(
                            waypoints_np, 
                            velocities_np, 
                            durations_np, 
                            yaw=0.0
                        )
                        
                        # Create ExecuteTrajectory command with 10Hz sample rate
                        full_trajectory_commands.append((agent_id, ExecuteTrajectory(trajectory_pieces, sample_rate=10)))
                    
                    # Update sequence with the full trajectory command
                    sequence[3] = full_trajectory_commands.copy()
                    print('   ✓ Full trajectory generation complete')
                else:
                    # This was the full trajectory execution
                    print(f'   ✓ Full trajectory execution complete for all drones')

            elif step_commands and type(step_commands[0][1]) is Land:
                print('   Landing complete for all drones')
            elif step_commands and type(step_commands[0][1]) is Arm:
                print('   Arming complete for all drones')
            
            step_idx += 1

    except KeyboardInterrupt:
        print('\n!!! KEYBOARD INTERRUPT DETECTED !!!')
        print('!!! SAFE LANDING ALL DRONES !!!')
        
        # Send land command to all drones
        for cf_id in range(num_agents):
            try:
                controlQueues[cf_id].put(Land(LAND_TIME))
                print(f'   Landing command sent to drone {cf_id}')
            except Exception as land_error:
                print(f'   Failed to send land command to drone {cf_id}: {land_error}')
        
        # Wait for landing to complete
        print(f'   Waiting {LAND_TIME} seconds for safe landing...')
        time.sleep(LAND_TIME + 1.0)
        print('   Safe landing complete')
        
    except Exception as e:
        print(f'\n!!! ERROR DETECTED: {type(e).__name__}: {e}')
        print('\n!!! FULL STACK TRACE:')
        traceback.print_exc()
        print('\n!!! EMERGENCY LANDING ALL DRONES !!!')
        
        # Send emergency land command to all drones
        for cf_id in range(num_agents):
            try:
                controlQueues[cf_id].put(Land(LAND_TIME))
                print(f'   Emergency land command sent to drone {cf_id}')
            except Exception as land_error:
                print(f'   Failed to send land command to drone {cf_id}: {land_error}')
        
        # Wait for landing to complete
        print(f'   Waiting {LAND_TIME} seconds for emergency landing...')
        time.sleep(LAND_TIME + 1.0)
        print('   Emergency landing complete')
        
    finally:
        # Always send Quit commands to cleanly exit drone control threads
        for ctrl in controlQueues:
            ctrl.put(Quit())
        
        # Generate plots after execution (for fixed trajectory mode)
        print('\n=== Generating plots ===')
        try:
            # Set matplotlib to use non-interactive backend (required when running in background thread)
            import matplotlib
            matplotlib.use('Agg')
            
            # Create unique timestamp directory for saving plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = Path(f"plots/live_demo/{timestamp}")
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f'Saving plots to: {plot_dir}')
            
            # Use the fixed trajectories for plotting
            x_trajs_np = np.array(fixed_x_trajs)
            goals_np = np.array(goals)
            init_positions_np = init_states[:, :pos_dim] if init_states is not None else np.zeros((num_agents, pos_dim))
            
            # 1. Plot basic trajectories (PNG)
            print('  Generating trajectory plot...')
            plot_drone_agent_trajs(
                x_trajs_np,
                goals_np,
                init_positions_np,
                title=f'Fixed Trajectory Execution (N={num_agents})',
                save_path=str(plot_dir / 'trajectories.png')
            )
            
            # 2. Generate trajectory GIF
            print('  Generating trajectory GIF...')
            plot_drone_agent_trajs_gif(
                x_trajs_np,
                goals_np,
                init_positions_np,
                save_path=str(plot_dir / 'trajectories.gif'),
                fps=10,
                title=f'Fixed Trajectory Execution (N={num_agents})'
            )
            
            # Save trajectory data for each agent
            print('\n=== Saving trajectory data ===')
            try:
                for agent_id in range(num_agents):
                    # Extract trajectory data: positions (px, py, pz) and velocities (vx, vy, vz)
                    positions = x_trajs_np[agent_id, :, :pos_dim]
                    velocities = x_trajs_np[agent_id, :, pos_dim:]
                    
                    # Combine trajectory data
                    trajectory_data = np.concatenate([positions, velocities], axis=1)
                    
                    # Save trajectory data as CSV
                    traj_filename = plot_dir / f'trajectory_agent{agent_id}.csv'
                    traj_header = 'px,py,pz,vx,vy,vz'
                    np.savetxt(traj_filename, trajectory_data, delimiter=',', header=traj_header, comments='')
                    print(f'  Saved trajectory data for agent {agent_id} to: {traj_filename}')
                
                print(f'\n✓ All trajectory data saved to: {plot_dir}')
                
            except Exception as save_error:
                print(f'\n!!! Error saving trajectory data: {save_error}')
                traceback.print_exc()
            
            print(f'\n✓ All plots saved to: {plot_dir}')
            
        except Exception as plot_error:
            print(f'\n!!! Error generating plots: {plot_error}')
            traceback.print_exc()


if __name__ == "__main__":
    num_agents = len(uris)
    
    # ==========================================
    # FIXED TRAJECTORY MODE
    # ==========================================
    # Specify the path to pre-computed trajectories
    # Supported formats:
    #   1. Single CSV file: agent_id,timestep,x,y,z,vx,vy,is_start,is_goal
    #      Example: plots/diffusion_trajs/two_agent_head_on/3D_trajectories.csv
    #   2. Directory with separate files: trajectory_agent0.csv, trajectory_agent1.csv, etc.
    #      Each CSV should have columns: px,py,pz,vx,vy,vz
    
    trajectory_path = config.get("fixed_trajectory_dir", None)
    if trajectory_path is None:
        raise ValueError("fixed_trajectory_dir must be specified in config for fixed trajectory mode")
    
    # Load the fixed trajectories
    print(f'\n=== Loading Fixed Trajectories ===')
    fixed_x_trajs = load_fixed_trajectories(trajectory_path, num_agents)
    T_total_traj = fixed_x_trajs.shape[1]
    
    print(f'Loaded {num_agents} agent trajectories with {T_total_traj} timesteps')
    
    # Simplified sequence: arm -> takeoff -> execute_full_trajectory -> land
    sequence = []
    arm_sequence = [(i, Arm()) for i in range(num_agents)]
    
    # Generate takeoff heights and create takeoff sequence
    use_custom_takeoff_heights = config.get("use_custom_takeoff_heights", False)
    if use_custom_takeoff_heights:
        print("Using custom takeoff heights from config")
        takeoff_heights = config.get("custom_takeoff_heights", None)
        if takeoff_heights is None:
            raise ValueError("custom_takeoff_heights is missing")
    else:
        takeoff_heights = [random.uniform(0.25, 1.5) for _ in range(num_agents)]
        # takeoff_heights = [1.0, 1.0]

    takeoff_sequence = [(i, Takeoff(takeoff_heights[i], STEP_TIME)) for i in range(num_agents)]
    land_sequence = [(i, Land(LAND_TIME)) for i in range(num_agents)]
    
    sequence.append(arm_sequence)
    sequence.append(takeoff_sequence)
    # Placeholder for fly-to-start trajectory (will be filled after takeoff)
    sequence.append([None] * num_agents)
    # Placeholder for full trajectory execution (will be filled after fly-to-start)
    sequence.append([None] * num_agents)
    sequence.append(land_sequence)

    controlQueues = [Queue() for _ in range(len(uris))]

    # Extract goal positions from the final positions in the fixed trajectories
    goals = []
    for agent_id in range(num_agents):
        final_pos = fixed_x_trajs[agent_id, -1, :3]  # Last position (px, py, pz)
        goals.append(final_pos)
    goals = jnp.array(goals)
    
    # Package optimization parameters (still needed for trajectory generation)
    opt_params = {
        'dt': dt,
        'planning_horizon': planning_horizon,
        'mask_horizon': mask_horizon,
        'mask_threshold': mask_threshold,
        'num_iters': num_iters,
        'step_size': step_size,
        'pos_dim': pos_dim,
        'u_dim': u_dim,
        'x_dim': x_dim
    }
    
    print(f'\n=== Configuration ===')
    print(f'  Mode: Fixed Trajectory Execution')
    print(f'  Trajectory path: {trajectory_path}')
    print(f'  Number of drones: {len(uris)}')
    print(f'  URIs: {uris}')
    print(f'  Sequence length: {len(sequence)} steps')
    print(f'  Takeoff heights: {takeoff_heights}')
    print(f'  Goal positions (from trajectory): {goals}')
    print(f'  Trajectory timesteps: {T_total_traj}')
    print(f'  dt: {dt}')
    print(f'  Total trajectory duration: {T_total_traj * dt:.2f}s')
    print(f'\nAttempting to connect to drones...')

    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        print('All drones connected successfully!')
        swarm.reset_estimators()

        print('\nStarting fixed trajectory execution!')

        threading.Thread(target=control_thread_fixed_trajectory, 
                        args=(swarm, fixed_x_trajs, goals, opt_params)).start()

        swarm.parallel_safe(crazyflie_control)

        time.sleep(1)
