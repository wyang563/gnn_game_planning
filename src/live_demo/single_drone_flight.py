import logging
import time
from typing import List, Tuple

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


LOGGER = logging.getLogger(__name__)

Point3D = Tuple[float, float, float]

URI: str = "radio://0/80/2M/E7E7E7E702"
START_POINT: Point3D = (0.0, 0.0, 0.3)
END_POINT: Point3D = (1.0, 0.0, 0.3)
WAYPOINTS: List[Point3D] = [(0.0, 0.0, 0.3), (1.0, 0.0, 0.3), (1.0, 0.0, 1.3), (1.0, 0.0, 0.3)]
DEFAULT_HEIGHT: float = 0.3
SPEED: float = 0.2


def fly_list_points(
    uri: str,
    points: List[Point3D],
    default_height: float = DEFAULT_HEIGHT,
    speed: float = SPEED,
) -> None:
    """
    Fly through a list of 3D points, pausing 1 second at each.

    The function assumes the Crazyflie starts at the first point in `points`
    in the MotionCommander coordinate frame.
    """
    if not points:
        LOGGER.warning("No waypoints provided, aborting flight")
        return

    LOGGER.info("Initializing CRTP drivers")
    cflib.crtp.init_drivers(enable_debug_driver=False)

    LOGGER.info("Connecting to Crazyflie at URI %s", uri)
    cf = Crazyflie(rw_cache="./cache")

    with SyncCrazyflie(uri, cf=cf) as scf:
        LOGGER.info("Connected, starting MotionCommander")
        with MotionCommander(scf, default_height=default_height) as mc:
            LOGGER.info("Starting multi-point flight through %d waypoints", len(points))

            # Assume we start at the first waypoint and hover briefly.
            current = points[0]
            LOGGER.info("Hovering at initial waypoint %s", current)
            time.sleep(1.0)

            for next_point in points[1:]:
                dx = next_point[0] - current[0]
                dy = next_point[1] - current[1]
                dz = next_point[2] - current[2]

                LOGGER.info(
                    "Flying from %s to %s (delta: dx=%.2f, dy=%.2f, dz=%.2f, v=%.2f)",
                    current,
                    next_point,
                    dx,
                    dy,
                    dz,
                    speed,
                )

                mc.move_distance(dx, dy, dz, velocity=speed)

                LOGGER.info("Reached waypoint %s, hovering for 1s", next_point)
                time.sleep(1.0)

                current = next_point

        LOGGER.info("Completed multi-point flight, landed and disconnected")


def fly_point_to_point(
    uri: str,
    start: Point3D,
    end: Point3D,
    default_height: float = 0.3,
    velocity: float = 0.2,
) -> None:
    LOGGER.info("Initializing CRTP drivers")
    cflib.crtp.init_drivers(enable_debug_driver=False)

    LOGGER.info("Connecting to Crazyflie at URI %s", uri)
    # rw_cache enables log/param table caching on disk which speeds up reconnects.
    cf = Crazyflie(rw_cache="./cache")

    with SyncCrazyflie(uri, cf=cf) as scf:
        LOGGER.info("Connected, starting MotionCommander")
        with MotionCommander(scf, default_height=default_height) as mc:
            # We are now hovering at approximately (0, 0, default_height)
            LOGGER.info("Hovering at default height %.2f m", default_height)

            # Compute the delta from start to end in the MotionCommander frame.
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]

            LOGGER.info(
                "Flying from %s to %s (delta: dx=%.2f, dy=%.2f, dz=%.2f, v=%.2f)",
                start,
                end,
                dx,
                dy,
                dz,
                velocity,
            )

            # Move in a straight line to the target point.
            mc.move_distance(dx, dy, dz, velocity=velocity)

            # Brief hover at the end point before landing.
            LOGGER.info("Reached target, hovering briefly before landing")
            time.sleep(1.0)

        LOGGER.info("Landed and disconnected")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    LOGGER.info("Start point: %s, end point: %s", START_POINT, END_POINT)

    try:
        # fly_point_to_point(
        #     uri=URI,
        #     start=START_POINT,
        #     end=END_POINT,
        #     default_height=DEFAULT_HEIGHT,
        #     velocity=SPEED,
        # )
        fly_list_points(
            uri=URI,
            points=WAYPOINTS,
            default_height=DEFAULT_HEIGHT,
            speed=SPEED,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user, attempting to land/clean up")
        # SyncCrazyflie + MotionCommander context managers handle cleanup
    except Exception:
        LOGGER.exception("Unexpected error during flight")
        raise


if __name__ == "__main__":
    main()
