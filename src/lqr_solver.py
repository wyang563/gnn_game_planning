import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List
from lqrax import iLQR

class Agent(iLQR):
    def __init__(
        self,
        id: int,
        dt: float,
        x_dim: int,
        u_dim: int,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        x0: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_traj: jnp.ndarray,
        device: str = "cpu",
    ) -> None:
        super().__init__(dt, x_dim, u_dim, Q, R)

        self.id: int = id
        self.x0: jnp.ndarray = x0
        self.u_traj: jnp.ndarray = u_traj
        self.ref_traj: jnp.ndarray = ref_traj  # shape (T, 2) for (x, y)
        self.x_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim)
        self.A_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim, x_dim)
        self.B_traj: Optional[jnp.ndarray] = None  # will be (T, x_dim, u_dim)

        self.jit_linearize_dyn = jit(self.linearize_dyn, device=device)
        self.jit_solve_ilqr = jit(self.solve, device=device)
        self.jit_loss = jit(self.loss, device=device)
        self.jit_linearize_loss = jit(self.linearize_loss, device=device)
    
    def dyn(self, xt: jnp.ndarray, ut: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([
            ut[0] * jnp.cos(xt[2]),
            ut[0] * jnp.sin(xt[2]),
            ut[1]
        ])
    
    def runtime_loss(
        self,
        xt: jnp.ndarray,
        ut: jnp.ndarray,
        ref_xt: jnp.ndarray,
        other_xts: jnp.ndarray,
    ) -> jnp.ndarray:
        nav_loss: jnp.ndarray = jnp.sum(jnp.square(xt[:2]-ref_xt[:2]))
        d2: jnp.ndarray = jnp.sum(jnp.square(xt[:2]-other_xts[:, :2]), axis=1)
        collision_loss: jnp.ndarray = jnp.sum(10.0 * jnp.exp(-5.0 * d2))
        ctrl_loss: jnp.ndarray = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.01])))
        return nav_loss + collision_loss + ctrl_loss

    def loss(
        self,
        x_traj: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_x_traj: jnp.ndarray,
        other_x_trajs: jnp.ndarray,
    ) -> jnp.ndarray:
        per_t = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0))
        return per_t(x_traj, u_traj, ref_x_traj, other_x_trajs).sum() * self.dt

    def linearize_loss(
        self,
        x_traj: jnp.ndarray,
        u_traj: jnp.ndarray,
        ref_x_traj: jnp.ndarray,
        other_x_trajs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj: jnp.ndarray = vmap(dldx, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)  # (T, x_dim)
        b_traj: jnp.ndarray = vmap(dldu, in_axes=(0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs)  # (T, u_dim)
        return a_traj, b_traj

class LQRSolver(iLQR):
    def __init__(self, 
                 n_agents: int = 2,
                 horizon: int = 30,
                 init_ps: List[jnp.ndarray] = None, # initial positions (x, y, theta)
                 init_us: List[jnp.ndarray] = None, # initial controls (v, omega)
                 goals: List[jnp.ndarray] = None,
                 dt: float = 0.1,
                 tsteps: int = 100,
                 Q: jnp.ndarray = jnp.eye(3), # penalties for deviation from ideal path
                 R: jnp.ndarray = jnp.eye(2), # penalties for deviation from ideal controls
                 device: str = "cpu"
                 ):
        self.n_agents: int = n_agents
        self.init_ps: List[jnp.ndarray] = init_ps
        self.init_us: List[jnp.ndarray] = init_us
        self.goals: List[jnp.ndarray] = goals
        self.horizon: int = horizon
        self.dt: float = dt
        self.tsteps: int = tsteps
        self.agents: List[Agent] = []
        self.device: str = device
        
        # create agents
        for i in range(n_agents):
            goal = self.goals[i]
            x0 = self.init_ps[i]
            u_traj = jnp.tile(self.init_us[i], reps=(tsteps, 1))
            ref_traj = jnp.linspace(x0[:2], goal, tsteps+1)[1:]
            agent = Agent(i, dt, x_dim=3, u_dim=2, Q=Q, R=R, x0=x0, u_traj=u_traj, ref_traj=ref_traj, device=device)
            self.agents.append(agent)
    
    def solve(self, num_iters: int = 200, step_size: float = 0.002):
        for iter in range(num_iters + 1):
            # get x, A, and B trajectories for each agent
            for agent in self.agents:
                agent.x_traj, agent.A_traj, agent.B_traj = agent.jit_linearize_dyn(
                    agent.x0, agent.u_traj   
                )
            
            for agent in self.agents:
                # get other trajectories
                other_x_trajs_list = [other_agent.x_traj for other_agent in self.agents if other_agent.id != agent.id]
                
                # Convert to JAX array for vectorized operations
                if len(other_x_trajs_list) == 0:
                    # No other agents - use dummy trajectory
                    other_x_trajs = jnp.zeros((self.tsteps, 1, 1))
                else:
                    # Stack multiple other trajectories
                    other_x_trajs = jnp.stack(other_x_trajs_list, axis=1)
                
                a_traj, b_traj = agent.jit_linearize_loss(
                    agent.x_traj, agent.u_traj, agent.ref_traj, other_x_trajs
                )
                v_traj, _ = agent.jit_solve_ilqr(
                    agent.A_traj, agent.B_traj, a_traj, b_traj
                )
                if iter % max(1, int(num_iters/10)) == 0:
                    loss = agent.jit_loss(
                        agent.x_traj, agent.u_traj, agent.ref_traj, other_x_trajs
                    )
                    print(f"Agent {agent.id}: loss = {loss:.6f}")
                agent.u_traj += step_size * v_traj

if __name__ == "__main__":
    # Example: 2 agents crossing paths
    print("Testing LQRSolver with 2 agents...")
    
    # Agent 0: Start at (-2, 0), goal at (2, 0) - moving right
    # Agent 1: Start at (2, 0), goal at (-2, 0) - moving left
    init_ps = [
        jnp.array([-2.0, 0.0, 0.0]),  # (x, y, theta)
        jnp.array([2.0, 0.0, jnp.pi])  # facing left
    ]
    
    init_us = [
        jnp.array([0.8, 0.0]),  # (v, omega) - forward motion
        jnp.array([0.8, 0.0])   # forward motion
    ]
    
    goals = [
        jnp.array([2.0, 0.0]),   # Agent 0 goal
        jnp.array([-2.0, 0.0])   # Agent 1 goal
    ]
    
    # Cost matrices
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.01]))  # State penalties
    R = jnp.diag(jnp.array([1.0, 0.01]))       # Control penalties
    
    # Create solver
    solver = LQRSolver(
        n_agents=2,
        init_ps=init_ps,
        init_us=init_us,
        goals=goals,
        dt=0.05,
        tsteps=100,
        Q=Q,
        R=R,
        device="cpu"
    )
    
    print(f"Created solver with {solver.n_agents} agents")
    print(f"Agent 0: start={init_ps[0][:2]}, goal={goals[0]}")
    print(f"Agent 1: start={init_ps[1][:2]}, goal={goals[1]}")
    
    # Solve the Nash equilibrium
    solver.solve(num_iters=200, step_size=0.002)
    
    print("\nSolution completed!")
    # print(f"Final positions:")
    # for i, (x_traj, u_traj) in enumerate(zip(x_trajs, u_trajs)):
    #     final_pos = x_traj[-1][:2]
    #     print(f"  Agent {i}: {final_pos}")
    
    # # Check if agents reached their goals
    # for i, (x_traj, goal) in enumerate(zip(x_trajs, goals)):
    #     final_pos = x_traj[-1][:2]
    #     distance = jnp.linalg.norm(final_pos - goal)
    #     print(f"Agent {i} distance to goal: {distance:.3f}")




            
            
            

