import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, Optional, List
from lqrax import LQR, iLQR

class Agent(iLQR):
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)
    
    def dyn(self, xt, ut):
        return jnp.array([
            ut[0] * jnp.cos(xt[2]),
            ut[0] * jnp.sin(xt[2]),
            ut[1]
        ])

class LQRSolver(iLQR):
    def __init__(self, 
                 n_agents: int = 2,
                 horizon: int = 30,
                 dt: float = 0.1,
                 weights: List[float, 4] = jnp.array([1.0, 1.0, 0.1, 0.0]),
                 speed_bound: float = 5.0,
                 acc_bound: float = 2.0
                 ):
        self.n_agents = n_agents
        self.horizon = horizon
        self.dt = dt
        self.w1, self.w2, self.w3, self.w4 = weights
        self.speed_bound = speed_bound
        self.acc_bound = acc_bound

        self.state_dim = 4
        self.control_dim = 2
