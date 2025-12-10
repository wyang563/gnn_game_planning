"""
Crazyflie Swarm Control with Step-by-Step Execution

This script controls one or more Crazyflie drones with manual step progression.
Commands are organized into steps that execute in parallel across drones.
After each Goto step, the system pauses and waits for user input.

SETUP:
1. Set ACTIVE_DRONES to the number of drones you want to use (1 or 2)
2. Set USE_TWO_DRONES = True/False in the main section to select sequence
3. Make sure ACTIVE_DRONES matches your sequence (1 for single, 2 for dual)

SEQUENCE FORMAT:
- List of steps: [step0, step1, step2, ...]
- Each step: [(drone_id, command), (drone_id, command), ...]
- All commands in a step execute simultaneously
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
from collections import namedtuple
from queue import Queue

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from utils.ops import Arm, Takeoff, Land, Goto, Ring, Quit
from utils.config_parser import parse_config

config = parse_config("src/live_demo/live_config.yaml")

# setup drone radio URIs
uris = []
for uri_num in config["drone_uris"]:
    uris.append(f"radio://0/80/2M/E7E7E7E7{uri_num}")

# DEFAULT_SPEED = config["default_speed"]
STEP_TIME = config["step_time"]
HOVER_TIME = 1000.0

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

def control_thread():
    stop = False
    step_idx = 0

    while not stop:
        if step_idx >= len(sequence):
            print('Reaching the end of the sequence, stopping!')
            stop = True
            break
        
        step_commands = sequence[step_idx]
        print(f'\nStep {step_idx}:')
        
        # Send all commands in this step to their respective drones
        max_duration = 0
        has_goto = False
        goto_drones = []
        
        for cf_id, command in step_commands:
            print(f' - Running: {command} on drone {cf_id}')
            controlQueues[cf_id].put(command)
            
            # Track the longest duration in this step
            if type(command) is Goto:
                max_duration = max(max_duration, command.time)
                has_goto = True
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
        
        # If there were Goto commands, maintain hover and wait for user input
        if has_goto:
            # Send long-duration hover commands to maintain position
            for cf_id, goto_cmd in goto_drones:
                controlQueues[cf_id].put(Goto(goto_cmd.x, goto_cmd.y, goto_cmd.z, HOVER_TIME))
                print(f'   Drone {cf_id} hovering at ({goto_cmd.x}, {goto_cmd.y}, {goto_cmd.z})')
            
            time.sleep(0.5)  # Brief delay for hover commands to take effect
            input('\n   All drones in position. Press Enter to proceed to next step...')
        else:
            # For non-Goto steps (Arm, Takeoff, Land), just report completion
            if step_commands and type(step_commands[0][1]) is Takeoff:
                print('   Takeoff complete for all drones')
            elif step_commands and type(step_commands[0][1]) is Land:
                print('   Landing complete for all drones')
            elif step_commands and type(step_commands[0][1]) is Arm:
                print('   Arming complete for all drones')
        
        step_idx += 1

    for ctrl in controlQueues:
        ctrl.put(Quit())

if __name__ == "__main__":
    sequence = [
        [(0, Arm()), (1, Arm())],
        [(0, Takeoff(0.5, STEP_TIME)), (1, Takeoff(1.0, STEP_TIME))],
        [(0, Goto(-0.5, -0.5, 0.5, STEP_TIME)), (1, Goto(0.5, 0.5, 1.0, STEP_TIME))],
        [(0, Goto(0.5, 0.5, 0.5, STEP_TIME)), (1, Goto(-0.5, -0.5, 1.0, STEP_TIME))],
        [(0, Goto(-0.5, -0.5, 0.5, STEP_TIME)), (1, Goto(0.5, 0.5, 1.0, STEP_TIME))],
        [(0, Goto(-0.5, -0.5, 1.5, STEP_TIME)), (1, Goto(0.5, 0.5, 1.5, STEP_TIME))],
        [(0, Land(2)), (1, Land(2))]
    ]

    controlQueues = [Queue() for _ in range(len(uris))]
    
    print(f'Configuration:')
    print(f'  Number of drones: {len(uris)}')
    print(f'  URIs: {uris}')
    print(f'  Sequence length: {len(sequence)} commands')
    print(f'\nAttempting to connect to drones...')

    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        print('All drones connected successfully!')
        swarm.reset_estimators()

        print('\nStarting sequence!')

        threading.Thread(target=control_thread).start()

        swarm.parallel_safe(crazyflie_control)

        time.sleep(1)


