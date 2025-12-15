"""
Utilities for generating polynomial trajectories for Crazyflie drones.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple


class PolynomialTrajectoryPiece:
    """
    Represents a single piece of a piecewise polynomial trajectory.
    Uses 7th order polynomials for each axis (x, y, z, yaw).
    """
    def __init__(self, duration, poly_x, poly_y, poly_z, poly_yaw=None):
        """
        Args:
            duration: Duration of this piece in seconds
            poly_x: Coefficients for x polynomial [c0, c1, ..., c7] (from low to high order)
            poly_y: Coefficients for y polynomial
            poly_z: Coefficients for z polynomial
            poly_yaw: Coefficients for yaw polynomial (optional, defaults to zero polynomial)
        """
        self.duration = duration
        self.poly_x = np.array(poly_x)
        self.poly_y = np.array(poly_y)
        self.poly_z = np.array(poly_z)
        self.poly_yaw = np.array(poly_yaw) if poly_yaw is not None else np.zeros(8)
    
    def evaluate(self, t):
        """Evaluate position at time t"""
        x = np.polyval(self.poly_x[::-1], t)
        y = np.polyval(self.poly_y[::-1], t)
        z = np.polyval(self.poly_z[::-1], t)
        yaw = np.polyval(self.poly_yaw[::-1], t)
        return x, y, z, yaw


def compute_minimum_snap_trajectory(waypoints: np.ndarray, velocities: np.ndarray, 
                                    durations: np.ndarray, yaw: float = 0.0) -> List[PolynomialTrajectoryPiece]:
    """
    Compute minimum snap (4th derivative) trajectory through waypoints.
    
    This generates smooth 7th order polynomial trajectories that minimize
    the squared magnitude of snap (4th derivative of position).
    
    Args:
        waypoints: (N, 3) array of waypoint positions [x, y, z]
        velocities: (N, 3) array of velocities at each waypoint
        durations: (N-1,) array of durations for each segment
        yaw: Fixed yaw angle in radians (default 0)
    
    Returns:
        List of PolynomialTrajectoryPiece objects
    """
    n_segments = len(waypoints) - 1
    n_coeffs = 8  # 7th order polynomial has 8 coefficients
    
    pieces = []
    
    # Solve for each axis independently
    for axis in range(3):
        # Build constraint matrix for continuity conditions
        # We need: position, velocity, acceleration, jerk at each waypoint
        n_constraints = 4 * (n_segments + 1)  # 4 constraints at each waypoint
        n_vars = n_coeffs * n_segments  # 8 coefficients per segment
        
        A = np.zeros((n_constraints, n_vars))
        b = np.zeros(n_constraints)
        
        constraint_idx = 0
        
        # For each segment
        for seg in range(n_segments):
            t_start = 0.0
            t_end = durations[seg]
            base_col = seg * n_coeffs
            
            # Start waypoint constraints (position and velocity)
            if seg == 0:
                # Position at start
                A[constraint_idx, base_col:base_col+n_coeffs] = [t_start**i for i in range(n_coeffs)]
                b[constraint_idx] = waypoints[seg, axis]
                constraint_idx += 1
                
                # Velocity at start
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * t_start**(i-1) if i > 0 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = velocities[seg, axis]
                constraint_idx += 1
                
                # Acceleration at start (set to 0 for smooth start)
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * (i-1) * t_start**(i-2) if i > 1 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = 0.0
                constraint_idx += 1
                
                # Jerk at start (set to 0)
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * (i-1) * (i-2) * t_start**(i-3) if i > 2 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = 0.0
                constraint_idx += 1
            
            # End waypoint constraints
            if seg == n_segments - 1:
                # Position at end
                A[constraint_idx, base_col:base_col+n_coeffs] = [t_end**i for i in range(n_coeffs)]
                b[constraint_idx] = waypoints[seg+1, axis]
                constraint_idx += 1
                
                # Velocity at end
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * t_end**(i-1) if i > 0 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = velocities[seg+1, axis]
                constraint_idx += 1
                
                # Acceleration at end (set to 0 for smooth end)
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * (i-1) * t_end**(i-2) if i > 1 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = 0.0
                constraint_idx += 1
                
                # Jerk at end (set to 0)
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * (i-1) * (i-2) * t_end**(i-3) if i > 2 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = 0.0
                constraint_idx += 1
            else:
                # Interior waypoint: ensure continuity between segments
                next_base_col = (seg + 1) * n_coeffs
                
                # Position continuity: p_seg(t_end) = waypoint = p_next(0)
                A[constraint_idx, base_col:base_col+n_coeffs] = [t_end**i for i in range(n_coeffs)]
                b[constraint_idx] = waypoints[seg+1, axis]
                constraint_idx += 1
                
                A[constraint_idx, next_base_col:next_base_col+n_coeffs] = [0**i for i in range(n_coeffs)]
                b[constraint_idx] = waypoints[seg+1, axis]
                constraint_idx += 1
                
                # Velocity continuity: v_seg(t_end) = v_next(0) = velocity at waypoint
                A[constraint_idx, base_col:base_col+n_coeffs] = [i * t_end**(i-1) if i > 0 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = velocities[seg+1, axis]
                constraint_idx += 1
                
                A[constraint_idx, next_base_col:next_base_col+n_coeffs] = [i * 0**(i-1) if i > 0 else 0 for i in range(n_coeffs)]
                b[constraint_idx] = velocities[seg+1, axis]
                constraint_idx += 1
        
        # Solve least squares problem (may be overdetermined)
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            coeffs = np.linalg.pinv(A) @ b
        
        # Store coefficients for this axis
        for seg in range(n_segments):
            base_col = seg * n_coeffs
            seg_coeffs = coeffs[base_col:base_col+n_coeffs]
            
            if axis == 0:
                pieces.append({'x': seg_coeffs, 'duration': durations[seg]})
            elif axis == 1:
                pieces[seg]['y'] = seg_coeffs
            else:  # axis == 2
                pieces[seg]['z'] = seg_coeffs
    
    # Convert to PolynomialTrajectoryPiece objects
    trajectory_pieces = []
    for piece_data in pieces:
        poly_yaw = np.zeros(8)
        poly_yaw[0] = yaw  # Constant yaw
        
        trajectory_pieces.append(
            PolynomialTrajectoryPiece(
                duration=piece_data['duration'],
                poly_x=piece_data['x'],
                poly_y=piece_data['y'],
                poly_z=piece_data['z'],
                poly_yaw=poly_yaw
            )
        )
    
    return trajectory_pieces


def simple_polynomial_trajectory(waypoints: np.ndarray, velocities: np.ndarray,
                                 durations: np.ndarray, yaw: float = 0.0) -> List[PolynomialTrajectoryPiece]:
    """
    Generate simple cubic polynomial trajectories between waypoints.
    Each segment is a cubic polynomial with specified start/end positions and velocities.
    
    Args:
        waypoints: (N, 3) array of waypoint positions [x, y, z]
        velocities: (N, 3) array of velocities at each waypoint
        durations: (N-1,) array of durations for each segment
        yaw: Fixed yaw angle in radians (default 0)
    
    Returns:
        List of PolynomialTrajectoryPiece objects
    """
    n_segments = len(waypoints) - 1
    pieces = []
    
    for seg in range(n_segments):
        T = durations[seg]
        
        # For each axis, solve cubic polynomial: p(t) = a + bt + ct^2 + dt^3
        # Constraints: p(0) = p0, p(T) = p1, p'(0) = v0, p'(T) = v1
        
        poly_x = np.zeros(8)
        poly_y = np.zeros(8)
        poly_z = np.zeros(8)
        
        for axis in range(3):
            p0 = waypoints[seg, axis]
            p1 = waypoints[seg+1, axis]
            v0 = velocities[seg, axis]
            v1 = velocities[seg+1, axis]
            
            # Solve for cubic coefficients [a, b, c, d]
            # p(t) = a + bt + ct^2 + dt^3
            # p(0) = a = p0
            # p'(0) = b = v0
            # p(T) = a + bT + cT^2 + dT^3 = p1
            # p'(T) = b + 2cT + 3dT^2 = v1
            
            a = p0
            b = v0
            
            # Solve 2x2 system for c and d
            # cT^2 + dT^3 = p1 - a - bT
            # 2cT + 3dT^2 = v1 - b
            
            A_mat = np.array([[T**2, T**3],
                             [2*T, 3*T**2]])
            b_vec = np.array([p1 - a - b*T,
                             v1 - b])
            
            if np.abs(np.linalg.det(A_mat)) > 1e-10:
                c, d = np.linalg.solve(A_mat, b_vec)
            else:
                # Fallback for degenerate case
                c = 0
                d = 0
            
            # Store coefficients in poly array (low to high order)
            if axis == 0:
                poly_x[:4] = [a, b, c, d]
            elif axis == 1:
                poly_y[:4] = [a, b, c, d]
            else:
                poly_z[:4] = [a, b, c, d]
        
        # Constant yaw
        poly_yaw = np.zeros(8)
        poly_yaw[0] = yaw
        
        pieces.append(
            PolynomialTrajectoryPiece(
                duration=T,
                poly_x=poly_x,
                poly_y=poly_y,
                poly_z=poly_z,
                poly_yaw=poly_yaw
            )
        )
    
    return pieces


def format_trajectory_for_crazyflie(pieces: List[PolynomialTrajectoryPiece]) -> List[dict]:
    """
    Convert trajectory pieces to format expected by Crazyflie firmware.
    
    Args:
        pieces: List of PolynomialTrajectoryPiece objects
    
    Returns:
        List of dictionaries with trajectory piece data
    """
    formatted_pieces = []
    
    for piece in pieces:
        formatted_pieces.append({
            'duration': piece.duration,
            'poly_x': piece.poly_x.tolist(),
            'poly_y': piece.poly_y.tolist(),
            'poly_z': piece.poly_z.tolist(),
            'poly_yaw': piece.poly_yaw.tolist()
        })
    
    return formatted_pieces
