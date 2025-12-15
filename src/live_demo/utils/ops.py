from collections import namedtuple

Arm = namedtuple('Arm', [])
Takeoff = namedtuple('Takeoff', ['height', 'time'])
Land = namedtuple('Land', ['time'])
Goto = namedtuple('Goto', ['x', 'y', 'z', 'time'])
# RGB [0-255], Intensity [0.0-1.0]
Ring = namedtuple('Ring', ['r', 'g', 'b', 'intensity', 'time'])
# Trajectory commands
UploadTrajectory = namedtuple('UploadTrajectory', ['trajectory_id', 'pieces'])
StartTrajectory = namedtuple('StartTrajectory', ['trajectory_id', 'timescale', 'relative'])
StopTrajectory = namedtuple('StopTrajectory', [])
# Reserved for the control loop, do not use in sequence
Quit = namedtuple('Quit', [])