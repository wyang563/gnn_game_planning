import sys
from pathlib import Path

# Add parent directory to path to allow importing from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
import traceback
from collections import namedtuple
from queue import Queue
import random
import jax.numpy as jnp
import jax

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from utils.ops import Arm, Takeoff, Land, Goto, Ring, Quit
from utils.config_parser import parse_config
from utils.init_goals import random_init
from models.train_gnn import load_trained_gnn_models
from solver.drone_agent import DroneAgent
from solver.solve import create_batched_loss_functions_mask
from load_config import load_config

config = parse_config("src/live_demo/live_config.yaml")
main_config = load_config()

# setup drone radio URIs
uris = []
for uri_num in config["drone_uris"]:
    uris.append(f"radio://0/80/2M/E7E7E7E7{uri_num}")

# DEFAULT_SPEED = config["default_speed"]
STEP_TIME = config["step_time"]
LAND_TIME = config["land_time"]
MIN_GOAL_DISTANCE = config["min_goal_separation_distance"]
XY_POSITION_RANGE = config["xy_position_range"]
Z_POSITION_RANGE = config["z_position_range"]
T_total = config["T_total"]
# T_total = 0

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
mask_threshold = main_config.testing.receding_horizon.mask_threshold

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
    control = controlQueues[uris.index(cf.link_uri)]

    activate_mellinger_controller(scf, False)

    commander = scf.cf.high_level_commander

    # Set fade to color effect and reset to Led-ring OFF
    cf.param.set_value('ring.effect', '14')

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
        elif type(command) is Goto:
            commander.go_to(command.x, command.y, command.z, 0, command.time)
        elif type(command) is Ring:
            # TODO: implement
            pass
        else:
            print('Warning! unknown command {} for uri {}'.format(command, cf.uri))

def get_player_masks(step_idx, x_trajs, model, model_state, mask_horizon, mask_threshold):
    # Extract past trajectories for mask calculation
    start_ind = max(0, step_idx - mask_horizon)
    end_ind = max(step_idx, 1)
    
    # Get past x_trajs: (n_agents, time_steps, x_dim)
    past_x_trajs = x_trajs[:, start_ind:end_ind, :]
    
    # Pad if we don't have enough history
    if past_x_trajs.shape[1] < mask_horizon:
        padding = jnp.tile(past_x_trajs[:, -1:, :], (1, mask_horizon - past_x_trajs.shape[1], 1))
        past_x_trajs = jnp.concatenate([past_x_trajs, padding], axis=1)
    
    # Transpose for GNN: (time_steps, n_agents, x_dim)
    past_x_trajs = past_x_trajs.transpose(1, 0, 2)
    
    # Add batch dimension: (1, time_steps, n_agents, x_dim)
    batch_past_x_trajs = past_x_trajs[None, ...]
    
    # Get masks from model
    masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
    masks = jnp.squeeze(masks, axis=0)  # squeeze batch dimension: (n_agents, n_agents)
    
    # Apply threshold to get binary masks
    masks = jnp.where(masks > mask_threshold, 1.0, 0.0)
    
    return masks

def solve_game(step_idx, player_masks, init_positions, ref_trajs, 
               agents, jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve,
               planning_horizon, num_iters, step_size, pos_dim, u_dim):
    n_agents = len(agents)
    
    # Setup horizon arrays
    next_x_trajs = None
    horizon_x0s = init_positions
    horizon_u_trajs = jnp.zeros((n_agents, planning_horizon, u_dim))
    
    # Extract reference trajectories for planning horizon
    start_ind = min(step_idx, ref_trajs.shape[1] - 1)
    end_ind = min(start_ind + planning_horizon, ref_trajs.shape[1])
    horizon_ref_trajs = ref_trajs[:, start_ind:end_ind, :]
    
    # Pad if planning horizon extends beyond reference trajectory
    if horizon_ref_trajs.shape[1] < planning_horizon:
        padding = jnp.tile(horizon_ref_trajs[:, -1:], (1, planning_horizon - horizon_ref_trajs.shape[1], 1))
        horizon_ref_trajs = jnp.concatenate([horizon_ref_trajs, padding], axis=1)
    
    # Game theory optimization loop
    for _ in range(num_iters + 1):
        # Linearize dynamics for all agents
        horizon_x_trajs, A_trajs, B_trajs = jit_batched_linearize_dyn(horizon_x0s, horizon_u_trajs)
        
        # Prepare other agent trajectories for collision avoidance
        # Create (n_agents, n_agents, planning_horizon, pos_dim) array
        all_x_pos = jnp.broadcast_to(horizon_x_trajs[None, :, :, :pos_dim], (n_agents, n_agents, planning_horizon, pos_dim))
        other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))  # (n_agents, planning_horizon, n_agents, pos_dim)
        
        # Tile masks for each timestep in horizon
        mask_for_step = jnp.tile(player_masks[:, None, :], (1, planning_horizon, 1))
        
        # Linearize loss for all agents
        a_trajs, b_trajs = jit_batched_linearize_loss(horizon_x_trajs, horizon_u_trajs, horizon_ref_trajs, other_x_trajs, mask_for_step)
        
        # Solve for optimal control updates
        v_trajs, _ = jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
        
        # Update control trajectories
        horizon_u_trajs += step_size * v_trajs
        next_x_trajs, _, _ = jit_batched_linearize_dyn(horizon_x0s, horizon_u_trajs)
    
    # Return the first control in the optimized horizon
    next_controls = horizon_u_trajs[:, 0, :]
    next_x_waypoints = next_x_trajs[:, 1, :]
    return next_controls, next_x_waypoints 

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

def control_thread(swarm, goals=None, model=None, model_state=None, 
                  agents=None, jit_batched_linearize_dyn=None, jit_batched_linearize_loss=None,
                  jit_batched_solve=None, opt_params=None):
    """
    Execute the sequence of commands step by step.
    
    Args:
        swarm: Swarm object to query drone positions
        goals: List of 3D goal positions [x, y, z] for each agent
        model: GNN model for player selection
        model_state: Model state with parameters
        agents: List of agent objects for optimization
        jit_batched_linearize_dyn: Batched dynamics linearization function
        jit_batched_linearize_loss: Batched loss linearization function
        jit_batched_solve: Batched solver function
        opt_params: Dictionary of optimization parameters
    """
    stop = False
    step_idx = 0
    num_agents = len(goals) if goals is not None else 0
    
    # Extract parameters
    u_dim = opt_params['u_dim']
    x_dim = opt_params['x_dim']
    
    # Initialize trajectory arrays (n_agents, T_total, dim)
    control_trajs = jnp.zeros((num_agents, T_total, u_dim))
    x_trajs = jnp.zeros((num_agents, T_total + 1, x_dim))  # +1 for initial state
    ref_trajs = None  # Will be initialized after takeoff
    init_states = None

    try:
        while not stop:
            if step_idx >= len(sequence):
                print('Reaching the end of the sequence, stopping!')
                stop = True
                break
            
            step_commands = sequence[step_idx]
            print(f'\nStep {step_idx}:')
            
            # Send all commands in this step to their respective drones
            max_duration = 0
            goto_drones = []
            
            for cf_id, command in step_commands:
                print(f' - Running: {command} on drone {cf_id}')
                controlQueues[cf_id].put(command)
                
                # Track the longest duration in this step
                if type(command) is Goto:
                    max_duration = max(max_duration, command.time)
                    goto_drones.append((cf_id, command))
                elif type(command) is Takeoff:
                    max_duration = max(max_duration, command.time)
                elif type(command) is Land:
                    max_duration = max(max_duration, command.time)
                elif type(command) is Arm:
                    max_duration = max(max_duration, 1.5)
            
            # Wait for all commands in this step to complete
            if max_duration > 0:
                time.sleep(max_duration + 0.5)
            
            # If there were Goto commands, compute next waypoint
            if step_commands and type(step_commands[0][1]) is Goto:
                for cf_id, goto_cmd in goto_drones:
                    print(f'   Drone {cf_id} hovering at ({goto_cmd.x}, {goto_cmd.y}, {goto_cmd.z})')
                
                # Calculate simulation timestep (account for arm and takeoff steps)
                sim_timestep = step_idx - 2  # First goto is at sequence[2], which is sim_timestep 0
                
                # Calculate trajectories up to current point using controls
                for agent_id in range(num_agents):
                    agent_x_traj, _, _ = agents[agent_id].linearize_dyn(init_states[agent_id], control_trajs[agent_id])
                    x_trajs = x_trajs.at[agent_id, :agent_x_traj.shape[0]].set(agent_x_traj)
                
                # Get current positions from trajectories
                current_positions = x_trajs[:, sim_timestep]
                
                print(f'   Computing optimal controls for timestep {sim_timestep}...')
                
                # Get player masks from GNN model
                player_masks = get_player_masks(sim_timestep, x_trajs, model, model_state, 
                                               opt_params['mask_horizon'], opt_params['mask_threshold'])
                
                # Solve for optimal next controls
                next_controls, next_x_trajs = solve_game(
                    sim_timestep,
                    player_masks,
                    current_positions,
                    ref_trajs,
                    agents,
                    jit_batched_linearize_dyn,
                    jit_batched_linearize_loss,
                    jit_batched_solve,
                    opt_params['planning_horizon'],
                    opt_params['num_iters'],
                    opt_params['step_size'],
                    opt_params['pos_dim'],
                    opt_params['u_dim'],
                )
                
                # Update control trajectory with new optimal controls
                control_trajs = control_trajs.at[:, sim_timestep, :].set(next_controls)
                
                # Create next goto commands from optimized trajectory
                # Use the first position from the optimized trajectory (index 1, since index 0 is current state)
                next_goto_sequence = []
                for agent_id in range(num_agents):
                    # Extract position from next state in optimized trajectory
                    next_pos = next_x_trajs[agent_id, :3]  # First 3 elements are [x, y, z]
                    
                    # Validate waypoint safety before sending command
                    validate_waypoint_safety(next_pos, agent_id, XY_POSITION_RANGE, Z_POSITION_RANGE)
                    
                    print(f'   Drone {agent_id} next waypoint: ({next_pos[0]:.3f}, {next_pos[1]:.3f}, {next_pos[2]:.3f})')
                    next_goto_sequence.append((agent_id, Goto(float(next_pos[0]), float(next_pos[1]), float(next_pos[2]), STEP_TIME)))
                
                # Update sequence with new goto commands for next step
                if step_idx + 1 < len(sequence) - 1:  # Don't overwrite land sequence
                    sequence[step_idx + 1] = next_goto_sequence

            elif step_commands and type(step_commands[0][1]) is Takeoff:
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
                    print(f'   Drone {uris.index(uri)} starting at position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]')
                
                # Initialize states with positions and zero velocities: [x, y, z, vx, vy, vz]
                init_states = jnp.array([[pos[0], pos[1], pos[2], 0.0, 0.0, 0.0] for pos in init_positions_list])
                
                # Set initial states in trajectory array
                x_trajs = x_trajs.at[:, 0, :].set(init_states)
                
                # Generate straight-line reference trajectories (position only)
                ref_trajs_list = []
                for agent_id in range(num_agents):
                    ref_traj = generate_straight_line_reference_trajectory(
                        init_positions_list[agent_id], goals[agent_id], T_total
                    )
                    ref_trajs_list.append(ref_traj)
                ref_trajs = jnp.array(ref_trajs_list)  # (n_agents, T_total, 3)
                
                # Compute first goto command
                print('   Computing initial optimal controls...')
                # For first step, use all agents (no history for masks yet, so use all-to-all)
                player_masks = jnp.ones((num_agents, num_agents)) - jnp.eye(num_agents)
                
                # Solve for first optimal controls
                first_controls, first_x_trajs = solve_game(
                    step_idx=0,
                    player_masks=player_masks,
                    init_positions=init_states,
                    ref_trajs=ref_trajs,
                    agents=agents,
                    jit_batched_linearize_dyn=jit_batched_linearize_dyn,
                    jit_batched_linearize_loss=jit_batched_linearize_loss,
                    jit_batched_solve=jit_batched_solve,
                    planning_horizon=opt_params['planning_horizon'],
                    num_iters=opt_params['num_iters'],
                    step_size=opt_params['step_size'],
                    pos_dim=opt_params['pos_dim'],
                    u_dim=opt_params['u_dim'],
                )
                
                # Update control trajectory
                control_trajs = control_trajs.at[:, 0, :].set(first_controls)
                
                # Create first goto commands from optimized trajectory
                # Use the first position from the optimized trajectory (index 1, since index 0 is current state)
                first_goto_sequence = []
                for agent_id in range(num_agents):
                    # Extract position from next state in optimized trajectory
                    next_pos = first_x_trajs[agent_id, :3]  # First 3 elements are [x, y, z]
                    
                    # Validate waypoint safety before sending command
                    validate_waypoint_safety(next_pos, agent_id, XY_POSITION_RANGE, Z_POSITION_RANGE)
                    
                    print(f'   Drone {agent_id} first waypoint: ({next_pos[0]:.3f}, {next_pos[1]:.3f}, {next_pos[2]:.3f})')
                    first_goto_sequence.append((agent_id, Goto(float(next_pos[0]), float(next_pos[1]), float(next_pos[2]), STEP_TIME)))
                
                # Update sequence with first goto commands (sequence[2])
                sequence[2] = first_goto_sequence
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


if __name__ == "__main__":
    num_agents = len(uris)
    
    # Initialize agents for game-theoretic optimization
    device = jax.devices()[0]
    agents = [DroneAgent(dt, x_dim=x_dim, u_dim=u_dim, Q=Q, R=R, 
                        collision_weight=collision_weight, collision_scale=collision_scale, 
                        ctrl_weight=control_weight, device=device) for _ in range(num_agents)]
    
    # Create batched loss functions for efficient computation
    for agent in agents:
        agent.create_loss_function_mask()
    
    # Create and set global batched functions
    jit_batched_linearize_dyn, jit_batched_linearize_loss, jit_batched_solve, _ = create_batched_loss_functions_mask(agents, device)
    
    sequence = []
    arm_sequence = [(i, Arm()) for i in range(num_agents)]
    
    # Generate takeoff heights and create takeoff sequence
    takeoff_heights = [random.uniform(0.25, 1.5) for _ in range(num_agents)]
    takeoff_sequence = [(i, Takeoff(takeoff_heights[i], STEP_TIME)) for i in range(num_agents)]
    
    land_sequence = [(i, Land(LAND_TIME)) for i in range(num_agents)]
    sequence.append(arm_sequence)
    sequence.append(takeoff_sequence)
    for _ in range(T_total):
        # set to None initially as placeholders (will be filled by optimization)
        goto_sequence = [None] * num_agents
        sequence.append(goto_sequence)
    sequence.append(land_sequence)

    controlQueues = [Queue() for _ in range(len(uris))]

    # generate random goals
    goals = random_init(num_agents, xy_position_range=[-1.5, 1.5], z_position_range=[0.25, 1.5], min_distance=MIN_GOAL_DISTANCE)

    # load GNN model for player selection
    model_path = "log/drone_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.001_bs_32_sigma1_0.11_sigma2_0.11_sigma3_0.25_noise_std_0.5_epochs_30_loss_type_similarity/20251203_144008/psn_best_model.pkl" 
    model_type = "gnn"
    model, model_state = load_trained_gnn_models(model_path, "full")
    
    # Package optimization parameters
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
    
    print(f'Configuration:')
    print(f'  Number of drones: {len(uris)}')
    print(f'  URIs: {uris}')
    print(f'  Sequence length: {len(sequence)} steps')
    print(f'  Takeoff heights: {takeoff_heights}')
    print(f'  Goals: {goals}')
    print(f'  Optimization parameters:')
    print(f'    dt: {dt}')
    print(f'    planning_horizon: {planning_horizon}')
    print(f'    mask_horizon: {mask_horizon}')
    print(f'    num_iters: {num_iters}')
    print(f'    step_size: {step_size}')
    print(f'\nAttempting to connect to drones...')

    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        print('All drones connected successfully!')
        swarm.reset_estimators()

        print('\nStarting sequence!')

        threading.Thread(target=control_thread, args=(swarm, goals, model, model_state, 
                                                      agents, jit_batched_linearize_dyn, 
                                                      jit_batched_linearize_loss, jit_batched_solve,
                                                      opt_params)).start()

        swarm.parallel_safe(crazyflie_control)

        time.sleep(1)


