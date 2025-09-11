import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
from typing import Tuple, Optional, List
from lqrax import LQR, iLQR

class Agent(iLQR):
    def __init__(
        self,
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
            agent = Agent(dt, x_dim=3, u_dim=2, Q=Q, R=R, x0=x0, u_traj=u_traj, ref_traj=ref_traj, device=device)
            self.agents.append(agent)
    
    def solve(self, num_iters: int = 200, step_size: float = 0.002):
        for iter in range(num_iters + 1):
            for agent in self.agents:
                agent.x_traj, agent.A_traj, agent.B_traj = agent.jit_linearize_dyn(
                    agent.x0, agent.u_traj   
                )
            
            # linearlize loss




            
            
            

