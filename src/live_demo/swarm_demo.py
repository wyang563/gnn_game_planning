"""
Crazyflie Swarm Control with Flexible Step Triggering

This script controls one or more Crazyflie drones with configurable step progression.
Commands are organized into steps that execute in parallel across drones.
After each Goto step, the system waits for a trigger signal before proceeding.

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

# TRIGGER FUNCTIONS
def create_timer_trigger(delay_seconds):
    """
    Create a timer-based trigger that waits a fixed duration between steps.
    
    Args:
        delay_seconds: How long to wait before proceeding to next step
    
    Returns:
        A wait function compatible with control_thread
    """
    def wait_function(step_idx, goto_drones):
        print(f'   Waiting {delay_seconds} seconds before next step...')
        time.sleep(delay_seconds)
    return wait_function


def create_event_trigger():
    """
    Create an event-based trigger that can be signaled externally.
    
    Example usage:
        wait_trigger, trigger_event = create_event_trigger()
        
        # In main thread: start drone control
        threading.Thread(target=control_thread, args=(wait_trigger,)).start()
        
        # In another thread/process: trigger next step
        trigger_event.set()  # Drones will proceed to next waypoint
    
    Returns:
        (wait_function, event): 
            - wait_function: Compatible with control_thread
            - event: threading.Event() that external code can set() to trigger next step
    """
    event = threading.Event()
    
    def wait_function(step_idx, goto_drones):
        print(f'   Waiting for external signal to proceed...')
        event.wait()  # Block until event.set() is called
        event.clear()  # Reset for next step
    
    return wait_function, event


def create_manual_trigger():
    """
    Create a manual input trigger (default behavior).
    
    Returns:
        A wait function compatible with control_thread
    """
    def wait_function(step_idx, goto_drones):
        input('\n   All drones in position. Press Enter to proceed to next step...')
    return wait_function


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

def control_thread(wait_for_next_step=None):
    """
    Execute the sequence of commands step by step.
    
    Args:
        wait_for_next_step: Optional callable that determines when to proceed to next step.
                           Signature: wait_for_next_step(step_idx, goto_drones) -> None
                           - step_idx: Current step number
                           - goto_drones: List of (drone_id, goto_command) tuples for this step
                           Should block until ready to proceed to next step.
                           If None, uses default input() prompt.
    """
    stop = False
    step_idx = 0
    
    # Default wait function: manual user input
    if wait_for_next_step is None:
        wait_for_next_step = create_manual_trigger()

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
        
        # If there were Goto commands, wait for signal to proceed
        if has_goto:
            for cf_id, goto_cmd in goto_drones:
                print(f'   Drone {cf_id} hovering at ({goto_cmd.x}, {goto_cmd.y}, {goto_cmd.z})')
            
            # Call the wait function (blocks until signal to proceed)
            wait_for_next_step(step_idx, goto_drones)
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
        [(0, Goto(-0.5, -0.5, 0.25, STEP_TIME)), (1, Goto(0.5, 0.5, 0.25, STEP_TIME))],
        [(0, Land(2)), (1, Land(2))]
    ]
    # sequence = [
    #     [(0, Arm())],
    #     [(0, Takeoff(0.25, STEP_TIME))],
    #     [(0, Goto(-0.5, -0.5, 0.25, STEP_TIME))],
    #     [(0, Goto(0.5, 0.5, 0.25, STEP_TIME))],
    #     [(0, Goto(-0.5, -0.5, 0.25, STEP_TIME))],
    #     [(0, Goto(-0.5, -0.5, 1.5, STEP_TIME))],
    #     [(0, Land(2))]
    # ]

    controlQueues = [Queue() for _ in range(len(uris))]
    
    # CONFIGURE TRIGGER FUNCTION HERE
    wait_trigger = create_timer_trigger(delay_seconds=3)
    
    print(f'Configuration:')
    print(f'  Number of drones: {len(uris)}')
    print(f'  URIs: {uris}')
    print(f'  Sequence length: {len(sequence)} steps')
    print(f'  Trigger mode: {wait_trigger.__name__ if hasattr(wait_trigger, "__name__") else "custom"}')
    print(f'\nAttempting to connect to drones...')

    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        print('All drones connected successfully!')
        swarm.reset_estimators()

        print('\nStarting sequence!')

        threading.Thread(target=control_thread, args=(wait_trigger,)).start()

        swarm.parallel_safe(crazyflie_control)

        time.sleep(1)


