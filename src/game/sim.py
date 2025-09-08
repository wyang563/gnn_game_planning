import torch
from .solver import NashSolver

class Simulator:
    def __init__(self, num_agents, region_size, solver_params, debug=False):
        self.num_agents = num_agents
        self.region_size = region_size
        self.dt = 0.1  
        
        self.positions = torch.rand((num_agents, 2)) * float(region_size)  # [x, y] for each agent
        self.velocities = torch.zeros((num_agents, 2))  # [vx, vy] for each agent
        
        self.targets = torch.rand((num_agents, 2)) * float(region_size)
        
        # initial positions
        if debug:
            print("targets")
            print(self.targets)
            print("initial positions")
            print(self.positions)
            
        self.initial_speed = 0.5
        self._set_initial_velocities()
        self.solver = NashSolver(Q_goal=solver_params['Q_goal'], Q_prox=solver_params['Q_prox'], R=solver_params['R'], safety_radius=solver_params['safety_radius'])
        self.trajectory_history = [[] for _ in range(num_agents)]
        self._update_trajectory_history()
    
    def _set_initial_velocities(self):
        for i in range(self.num_agents):
            direction = self.targets[i] - self.positions[i]
            distance = torch.norm(direction)
            self.velocities[i] = direction / distance * self.initial_speed
    
    def _update_trajectory_history(self):
        for i in range(self.num_agents):
            self.trajectory_history[i].append({
                'position': self.positions[i].clone(),
                'velocity': self.velocities[i].clone(),
                'timestamp': len(self.trajectory_history[i]) * self.dt
            })
    
    def call(self):
        try:
            # Call NashSolver to get optimal controls
            # Note: solve_nash_eq method needs to be implemented in NashSolver
            controls = self.solver.solve_nash_eq(
                positions=self.positions,
                velocities=self.velocities, 
                targets=self.targets,
                dt=self.dt
            )
            
            # If solver returns None or fails, use simple proportional control towards target
            if controls is None:
                controls = self._fallback_control()
                
            return controls
            
        except Exception as e:
            print(f"Nash solver failed, using fallback control: {e}")
            return self._fallback_control()
    
    def _fallback_control(self):
        controls = torch.zeros((self.num_agents, 2))
        
        for i in range(self.num_agents):
            position_error = self.targets[i] - self.positions[i]
            velocity_error = -self.velocities[i]  
            
            kp = 1.0  # Position gain
            kd = 0.5  # Velocity gain
            
            controls[i] = kp * position_error + kd * velocity_error
            
            # Limit acceleration magnitude (typical constraint)
            max_accel = 2.0  
            norm_val = torch.norm(controls[i])
            if float(norm_val) > max_accel:
                controls[i] = controls[i] / norm_val * max_accel
                
        return controls
    
    def step(self, controls):
        """
        Update agent states using double integrator dynamics.
        
        Based on the PSN Game paper (Section 3.1), agents follow double integrator dynamics:
        x_{k+1} = x_k + v_k * dt + 0.5 * u_k * dt^2
        v_{k+1} = v_k + u_k * dt
        
        Args:
            controls (torch.Tensor): Control inputs for each agent, shape (num_agents, 2)
        """
        if tuple(controls.shape) != (self.num_agents, 2):
            raise ValueError(f"Controls shape {controls.shape} doesn't match expected {(self.num_agents, 2)}")
        
        self.positions += self.velocities * self.dt + 0.5 * controls * (self.dt ** 2)
        self.velocities += controls * self.dt
        # self.positions = torch.clamp(self.positions, 0.0, float(self.region_size))
        
        max_velocity = 3.0  
        for i in range(self.num_agents):
            vel_norm = torch.norm(self.velocities[i])
            if float(vel_norm) > max_velocity:
                self.velocities[i] = self.velocities[i] / vel_norm * max_velocity
        self._update_trajectory_history()
    
    def get_states(self):
        """
        Get current states of all agents.
        
        Returns:
            dict: Dictionary containing positions, velocities, and targets
        """
        return {
            'positions': self.positions.clone(),
            'velocities': self.velocities.clone(), 
            'targets': self.targets.clone(),
            'time': len(self.trajectory_history[0]) * self.dt
        }
    
    def reset(self):
        """Reset simulation to initial random state."""
        self.positions = torch.rand((self.num_agents, 2)) * float(self.region_size)
        self.targets = torch.rand((self.num_agents, 2)) * float(self.region_size)
        
        self.velocities = torch.zeros((self.num_agents, 2))
        self._set_initial_velocities()
        
        self.trajectory_history = [[] for _ in range(self.num_agents)]
        self._update_trajectory_history()
