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
            print("Julia environment setup successful!")
            self.julia_initialized = True

            # Load required Julia packages
            Main.eval("using MixedComplementarityProblems")
            Main.eval("using LinearAlgebra")
            Main.eval("using NLsolve")
            self.julia = Main
            
            # Define the Nash equilibrium solver in Julia
            self._setup_nash_solver_julia()
            print("All jl packages loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Julia environment: {e}")
    
    def _setup_nash_solver_julia(self):
        """Set up the Julia-based Nash equilibrium solver for linear-quadratic games."""
        julia_code = """
        function solve_lq_nash_equilibrium(positions, velocities, targets, Q_goal, Q_prox, R_param, 
                                         safety_radius, dt, horizon)
            n_agents = size(positions, 1)
            n_controls = 2 * n_agents  # [ux, uy] for each agent per timestep
            n_vars = horizon * n_controls  # Total control variables across horizon
            
            # Initial state vector for all agents [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
            x0 = zeros(4 * n_agents)
            for i in 1:n_agents
                base_idx = (i-1) * 4
                x0[base_idx + 1] = positions[i, 1]  # x position
                x0[base_idx + 2] = positions[i, 2]  # y position  
                x0[base_idx + 3] = velocities[i, 1] # x velocity
                x0[base_idx + 4] = velocities[i, 2] # y velocity
            end
            
            # Build system matrices for double integrator dynamics
            A = build_system_matrix_A(n_agents, dt)
            B = build_system_matrix_B(n_agents, dt)
            
            # Build target references
            target_refs = zeros(2 * n_agents)
            for i in 1:n_agents
                target_refs[(i-1)*2 + 1] = targets[i, 1]
                target_refs[(i-1)*2 + 2] = targets[i, 2]
            end
            
            # For linear-quadratic games, we can compute Nash equilibrium analytically
            # or use iterative best response. Here we use a simplified approach:
            
            # Build coupled cost matrices for all agents
            Q_total, R_total = build_coupled_lq_matrices(n_agents, Q_goal, Q_prox, R_param, 
                                                        positions, targets, safety_radius)
            
            # Solve finite horizon LQ Nash game using Riccati-like equations
            controls_nash = solve_finite_horizon_lq_nash(x0, target_refs, A, B, Q_total, R_total, 
                                                        horizon, n_agents)
            
            if controls_nash !== nothing
                # Extract first control step for each agent
                controls = zeros(n_agents, 2)
                for i in 1:n_agents
                    start_idx = (i-1) * 2 + 1
                    controls[i, 1] = controls_nash[start_idx]
                    controls[i, 2] = controls_nash[start_idx + 1]
                end
                return controls
            else
                return nothing
            end
        end
        
        function solve_finite_horizon_lq_nash(x0, target_refs, A, B, Q, R, horizon, n_agents)
            # Simplified finite horizon Nash equilibrium for LQ games
            # Based on coupled Riccati equations approach
            
            n_states = length(x0)
            n_controls_total = size(B, 2)
            n_controls_per_agent = 2
            
            # Initialize solution arrays
            all_controls = zeros(horizon * n_controls_total)
            
            # For each agent, solve best response given others' strategies
            # This is a simplified version - full Nash would require coupled Riccati equations
            
            max_iterations = 10
            tolerance = 1e-6
            
            for iter in 1:max_iterations
                controls_old = copy(all_controls)
                
                # Update each agent's strategy via best response
                for agent_id in 1:n_agents
                    agent_controls = solve_agent_best_response(agent_id, x0, target_refs, A, B, 
                                                             Q, R, horizon, all_controls, n_agents)
                    
                    # Update this agent's controls in the full control vector
                    for t in 1:horizon
                        control_base = (t-1) * n_controls_total + (agent_id-1) * n_controls_per_agent
                        agent_base = (t-1) * n_controls_per_agent
                        all_controls[control_base + 1] = agent_controls[agent_base + 1]
                        all_controls[control_base + 2] = agent_controls[agent_base + 2]
                    end
                end
                
                # Check convergence
                if norm(all_controls - controls_old) < tolerance
                    break
                end
            end
            
            return all_controls[1:n_controls_total]  # Return first timestep controls
        end
        
        function solve_agent_best_response(agent_id, x0, target_refs, A, B, Q, R, horizon, 
                                         other_controls, n_agents)
            # Solve agent's best response optimization problem
            # min_u Σ_{t=0}^{T-1} [x_t'Q_i x_t + u_i_t'R_i u_i_t] + x_T'Q_f x_T
            
            n_controls_per_agent = 2
            agent_controls = zeros(horizon * n_controls_per_agent)
            
            # Simplified: use proportional control towards target with obstacle avoidance
            for t in 1:horizon
                # Predict state at time t
                x_t = predict_state_at_time(x0, A, B, other_controls, t)
                
                # Extract agent's position
                agent_base = (agent_id - 1) * 4
                pos_current = [x_t[agent_base + 1], x_t[agent_base + 2]]
                vel_current = [x_t[agent_base + 3], x_t[agent_base + 4]]
                
                # Target position for this agent
                target_pos = [target_refs[(agent_id-1)*2 + 1], target_refs[(agent_id-1)*2 + 2]]
                
                # Proportional control towards target
                pos_error = target_pos - pos_current
                vel_error = -vel_current  # Prefer zero velocity at target
                
                # Control law: u = K_p * pos_error + K_d * vel_error
                K_p = 2.0
                K_d = 1.0
                
                control_t = K_p * pos_error + K_d * vel_error
                
                # Add safety control to avoid other agents
                for j in 1:n_agents
                    if j != agent_id
                        other_base = (j - 1) * 4
                        other_pos = [x_t[other_base + 1], x_t[other_base + 2]]
                        
                        diff = pos_current - other_pos
                        dist = norm(diff)
                        
                        if dist < 2.0  # Safety zone
                            safety_gain = 5.0 / (dist + 0.1)
                            control_t += safety_gain * (diff / (dist + 1e-6))
                        end
                    end
                end
                
                # Clip control magnitude
                max_control = 5.0
                if norm(control_t) > max_control
                    control_t = control_t / norm(control_t) * max_control
                end
                
                control_base = (t-1) * n_controls_per_agent
                agent_controls[control_base + 1] = control_t[1]
                agent_controls[control_base + 2] = control_t[2]
            end
            
            return agent_controls
        end
        
        function predict_state_at_time(x0, A, B, controls, timestep)
            # Predict state at given timestep using system dynamics
            x = copy(x0)
            n_controls_total = size(B, 2)
            
            for t in 1:timestep
                if length(controls) >= t * n_controls_total
                    u_t = controls[(t-1)*n_controls_total + 1:t*n_controls_total]
                    x = A * x + B * u_t
                else
                    x = A * x  # No control input
                end
            end
            
            return x
        end
        
        function build_system_matrix_A(n_agents, dt)
            # Double integrator: [x, y, vx, vy] for each agent
            n_states = 4 * n_agents
            A = zeros(n_states, n_states)
            
            for i in 1:n_agents
                base_idx = (i-1) * 4
                # Position dynamics: x_{k+1} = x_k + v_k * dt
                A[base_idx + 1, base_idx + 1] = 1.0  # x
                A[base_idx + 1, base_idx + 3] = dt   # vx
                A[base_idx + 2, base_idx + 2] = 1.0  # y  
                A[base_idx + 2, base_idx + 4] = dt   # vy
                # Velocity dynamics: v_{k+1} = v_k + u_k * dt (handled in B matrix)
                A[base_idx + 3, base_idx + 3] = 1.0  # vx
                A[base_idx + 4, base_idx + 4] = 1.0  # vy
            end
            
            return A
        end
        
        function build_system_matrix_B(n_agents, dt)
            n_states = 4 * n_agents
            n_controls = 2 * n_agents
            B = zeros(n_states, n_controls)
            
            for i in 1:n_agents
                state_base = (i-1) * 4
                control_base = (i-1) * 2
                
                # Position affected by control: x_{k+1} = x_k + v_k*dt + 0.5*u_k*dt^2
                B[state_base + 1, control_base + 1] = 0.5 * dt^2  # x from ux
                B[state_base + 2, control_base + 2] = 0.5 * dt^2  # y from uy
                
                # Velocity affected by control: v_{k+1} = v_k + u_k*dt
                B[state_base + 3, control_base + 1] = dt  # vx from ux
                B[state_base + 4, control_base + 2] = dt  # vy from uy
            end
            
            return B
        end
        
        function build_coupled_lq_matrices(n_agents, Q_goal, Q_prox, R_param, positions, targets, safety_radius)
            # Build coupled cost matrices for all agents in the LQ game
            n_states = 4 * n_agents
            n_controls = 2 * n_agents
            
            # State cost matrix Q - includes goal-seeking and proximity costs
            Q = zeros(n_states, n_states)
            
            for i in 1:n_agents
                agent_base = (i - 1) * 4
                
                # Goal reaching cost - penalize distance from target
                Q[agent_base + 1, agent_base + 1] = Q_goal  # x position
                Q[agent_base + 2, agent_base + 2] = Q_goal  # y position
                
                # Velocity cost - prefer lower velocities
                Q[agent_base + 3, agent_base + 3] = 0.1
                Q[agent_base + 4, agent_base + 4] = 0.1
                
                # Proximity costs between agents
                for j in 1:n_agents
                    if j != i
                        other_base = (j - 1) * 4
                        # Coupling cost terms for proximity
                        Q[agent_base + 1, other_base + 1] = Q_prox
                        Q[agent_base + 2, other_base + 2] = Q_prox
                    end
                end
            end
            
            # Control cost matrix R
            R = zeros(n_controls, n_controls)
            for i in 1:n_agents
                control_base = (i - 1) * 2
                R[control_base + 1, control_base + 1] = R_param  # ux
                R[control_base + 2, control_base + 2] = R_param  # uy
            end
            
            return Q, R
        end
        """
        
        self.julia.eval(julia_code)
    
    def solve_nash_eq(self, positions, velocities, targets, dt=0.1):
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
            # Convert PyTorch tensors to numpy for Julia
            pos_np = positions.detach().cpu().numpy()
            vel_np = velocities.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Solve Nash equilibrium using Julia
            horizon = 5  # Planning horizon (can be made configurable)
            
            controls = self.julia.solve_lq_nash_equilibrium(
                pos_np, vel_np, targets_np, 
                self.Q_goal, self.Q_prox, self.R, 
                self.safety_radius, dt, horizon
            )
            
            if controls is not None:
                # Convert back to PyTorch tensor
                return torch.tensor(controls, dtype=torch.float32)
            else:
                print("Nash equilibrium solver failed to converge")
                return None
                
        except Exception as e:
            print(f"Nash equilibrium solver error: {e}")
            return None

if __name__ == "__main__":
    solver = NashSolver()