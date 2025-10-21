import jax.numpy as jnp
from lqrax import iLQR
from jax import vmap, grad, jit, debug
import jax

class PointAgent(iLQR):
    def __init__(self, dt, x_dim, u_dim, Q, R, collision_weight, collision_scale, ctrl_weight, device):
        self.collision_weight = collision_weight
        self.collision_scale = collision_scale
        self.ctrl_weight = ctrl_weight
        self.device = device
        super().__init__(dt, x_dim, u_dim, Q, R)
    
    def dyn(self, x, u):
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])

    def create_loss_functions_train(self):
        # TODO: Implement loss functions for training
        pass

    def create_loss_functions_test(self):
        def runtime_loss(xt, ut, ref_xt, other_states):
            nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))

            squared_distances = jnp.sum(jnp.square(xt[:2] - other_states[:, :2]), axis=1)
            collision_loss = self.collision_weight * jnp.exp(-self.collision_scale * squared_distances)
            collision_loss = jnp.sum(collision_loss)

            ctrl_loss = self.ctrl_weight * jnp.sum(jnp.square(ut))
            return nav_loss + collision_loss + ctrl_loss
        
        def trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            def single_step_loss(args):
                xt, ut, ref_xt, other_xts = args
                return runtime_loss(xt, ut, ref_xt, other_xts)
            loss_array = vmap(single_step_loss)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return loss_array.sum() * self.dt

        def linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            dldx = grad(runtime_loss, argnums=(0))
            dldu = grad(runtime_loss, argnums=(1))
            
            def grad_step(args):
                xt, ut, ref_xt, other_xts = args
                return dldx(xt, ut, ref_xt, other_xts), dldu(xt, ut, ref_xt, other_xts)
            
            grads = vmap(grad_step)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return grads[0], grads[1]  # a_traj, b_traj 
        
        self.runtime_loss = runtime_loss
        self.trajectory_loss = trajectory_loss
        self.linearize_loss = linearize_loss
        self.compiled_trajectory_loss = jit(trajectory_loss, device=self.device)
        self.compiled_linearize_loss = jit(linearize_loss, device=self.device)
        self.compiled_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.compiled_solve = jit(self.solve, device=self.device)
