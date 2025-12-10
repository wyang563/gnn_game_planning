import sys
from pathlib import Path

# Add parent directory to path to allow importing from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
from collections import namedtuple
from queue import Queue
import random
import jax.numpy as jnp

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from utils.ops import Arm, Takeoff, Land, Goto, Ring, Quit
from utils.config_parser import parse_config
from utils.init_goals import random_init
from models.train_gnn import load_trained_gnn_models

config = parse_config("src/live_demo/live_config.yaml")

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
# T_total = config["T_total"]
T_total = 0

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

def get_player_masks(past_trajs, model, model_state):
    start_ind = max(0, len(past_trajs) - 10)
    end_ind = max(start_ind, 1)
    past_x_trajs = past_trajs[start_ind:end_ind, :, :]
    if past_x_trajs.shape[0] < 10:
        padding = jnp.tile(past_x_trajs[-1:], (10 - past_x_trajs.shape[0], 1, 1))
        past_x_trajs = jnp.concatenate([padding, past_x_trajs], axis=0)
    batch_past_x_trajs = past_x_trajs[None, ...]

    masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
    masks = jnp.squeeze(masks, axis=0) # squeeze batch dimension
    return masks

def solve_game():
    pass

def control_thread(swarm, goals=None, model=None, model_state=None):
    """
    Execute the sequence of commands step by step.
    
    Args:
        swarm: Swarm object to query drone positions
        goals: List of 3D goal positions [x, y, z] for each agent
        takeoff_heights: List of takeoff heights for each agent
    """
    stop = False
    step_idx = 0
    past_trajs = jnp.zeros((T_total, num_agents, 6))
    ref_trajs = None  # Will be initialized after takeoff
    num_agents = len(goals) if goals is not None else 0

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
            
            # If there were Goto commands, wait for signal to proceed
            if step_commands and type(step_commands[0][1]) is Goto:
                for cf_id, goto_cmd in goto_drones:
                    print(f'   Drone {cf_id} hovering at ({goto_cmd.x}, {goto_cmd.y}, {goto_cmd.z})')
                
                player_masks = get_player_masks(past_trajs)


            elif step_commands and type(step_commands[0][1]) is Takeoff:
                print('   Takeoff complete for all drones')
                
                # Get absolute positions of each drone directly from swarm
                positions_dict = swarm.get_estimated_positions()
                current_positions = []
                for uri in uris:
                    pos = positions_dict.get(uri)
                    if pos:
                        position = [pos.x, pos.y, pos.z]
                    else:
                        raise ValueError(f'Position not available for drone {uri}')
                    current_positions.append(position)
                    print(f'   Drone {uris.index(uri)} starting at position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]')
                
                # Update past_trajs with initial positions after takeoff
                past_trajs = past_trajs.at[0, :, :3].set(current_positions)
                
                # Generate straight-line reference trajectories
                ref_trajs = []
                for agent_id in range(num_agents):
                    ref_traj = generate_straight_line_reference_trajectory(
                        past_trajs[0][agent_id], goals[agent_id], T_total
                    )
                    ref_trajs.append(ref_traj)
                ref_trajs = jnp.array(ref_trajs)
            elif step_commands and type(step_commands[0][1]) is Land:
                print('   Landing complete for all drones')
            elif step_commands and type(step_commands[0][1]) is Arm:
                print('   Arming complete for all drones')
            
            step_idx += 1

    except Exception as e:
        print(f'\n!!! ERROR DETECTED: {type(e).__name__}: {e}')
        print('!!! EMERGENCY LANDING ALL DRONES !!!')
        
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
    sequence = []
    arm_sequence = [(i, Arm()) for i in range(num_agents)]
    
    # Generate takeoff heights and create takeoff sequence
    takeoff_heights = [random.uniform(0.25, 1.5) for _ in range(num_agents)]
    takeoff_sequence = [(i, Takeoff(takeoff_heights[i], STEP_TIME)) for i in range(num_agents)]
    
    land_sequence = [(i, Land(LAND_TIME)) for i in range(num_agents)]
    sequence.append(arm_sequence)
    sequence.append(takeoff_sequence)
    for _ in range(T_total):
        # set to None initially as placeholders
        goto_sequence = [None] * num_agents
        sequence.append(goto_sequence)
    sequence.append(land_sequence)

    controlQueues = [Queue() for _ in range(len(uris))]

    # generate random goals
    goals = random_init(num_agents, xy_position_range=[-1.5, 1.5], z_position_range=[0.25, 1.5], min_distance=MIN_GOAL_DISTANCE)

    # load model
    model_path = None
    model_type = "gnn"
    model, model_state = load_trained_gnn_models(model_path, "full")
    
    print(f'Configuration:')
    print(f'  Number of drones: {len(uris)}')
    print(f'  URIs: {uris}')
    print(f'  Sequence length: {len(sequence)} steps')
    print(f'  Takeoff heights: {takeoff_heights}')
    print(f'  Goals: {goals}')
    print(f'\nAttempting to connect to drones...')

    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        print('All drones connected successfully!')
        swarm.reset_estimators()

        print('\nStarting sequence!')

        threading.Thread(target=control_thread, args=(swarm, goals, model, model_state)).start()

        swarm.parallel_safe(crazyflie_control)

        time.sleep(1)


