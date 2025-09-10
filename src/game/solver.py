import numpy as np
import torch

class NashSolver:
    def __init__(self, Q_goal, Q_prox, R, safety_radius):
        self.setup_julia()
        
        # Game parameters from PSN Game paper Section 3 and Appendix
        self.Q_goal = Q_goal      # Goal-reaching weight
        self.Q_prox = Q_prox     # Proximity cost weight  
        self.R = R           # Control cost weight
        self.safety_radius = safety_radius # Safety radius between agents
        
    def setup_julia(self):
        try:
            from julia import Main
            from julia import Pkg
            
            # Always setup project directory in Julia (needed for each Python process)
            project_dir = "/home/alex/gnn_game_planning"
            Main.eval(f'ENV["JULIA_PROJECT"] = "{project_dir}"')
            Pkg.activate(project_dir)
            
            Pkg.add("MixedComplementarityProblems")
            Pkg.add("LinearAlgebra")
            Pkg.add("NLsolve")
            Pkg.add("ForwardDiff")   
            print("Julia environment setup successful!")
            self.julia_initialized = True

            # Load required Julia packages
            Main.eval("using MixedComplementarityProblems")
            Main.eval("using LinearAlgebra")
            Main.eval("using NLsolve")
            Main.eval("using ForwardDiff")
            self.julia = Main
            
            # Define the Nash equilibrium solver in Julia
            self._setup_nash_solver_julia()
            print("All jl packages loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Julia environment: {e}")
    
    def solve_nash_eq(self, positions, velocities, targets, horizon=5,dt=0.1):
        """
        Solve Nash equilibrium for linear-quadratic multi-agent game.
        
        Args:
            positions (torch.Tensor): Current positions of all agents [n_agents, 2]
            velocities (torch.Tensor): Current velocities of all agents [n_agents, 2]  
            targets (torch.Tensor): Target positions for all agents [n_agents, 2]
            dt (float): Time step
            
        Returns:
            torch.Tensor: Nash equilibrium control inputs [n_agents, 2] or None if failed
        """
        try:
            pos_np = positions.detach().cpu().numpy()
            vel_np = velocities.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Solve Nash equilibrium using Julia
            controls = self.julia.solve_lq_nash_equilibrium(
                pos_np, vel_np, targets_np, 
                self.Q_goal, self.Q_prox, self.R, 
                self.safety_radius, dt, horizon
            )
            
            if controls is not None:
                return torch.tensor(controls, dtype=torch.float32)
            else:
                print("Nash equilibrium solver failed to converge")
                return None
                
        except Exception as e:
            print(f"Nash equilibrium solver error: {e}")
            return None

if __name__ == "__main__":
    solver = NashSolver()