import jax.numpy as jnp
from lqrax import iLQR
from jax import vmap, grad, jit, debug
import jax

class DroneAgent(iLQR):
    def __init__(self, dt, x_dim, u_dim, Q, R, collision_weight, collision_scale, ctrl_weight, device):
        self.collision_weight = collision_weight
        self.collision_scale = collision_scale
        self.ctrl_weight = ctrl_weight
        self.device = device
        self.g = 9.81
        super().__init__(dt, x_dim, u_dim, Q, R)
    
    def dyn(self, x, u):
        return jnp.array([
            x[3],  # dx/dt = vx
            x[4],  # dy/dt = vy
            x[5],  # dz/dt = vz
            u[0],  # dvx/dt = ax
            u[1],  # dvy/dt = ay
            u[2] - self.g
        ])

    def create_loss_function_mask(self):
        def runtime_loss(xt, ut, ref_xt, other_states, mask):
            nav_loss = jnp.sum(jnp.square(xt[:3] - ref_xt[:3]))

            squared_distances = jnp.sum(jnp.square(xt[:3] - other_states[:, :3]), axis=1)
            collision_loss = self.collision_weight * jnp.exp(-self.collision_scale * squared_distances)
            # apply mask to collision loss
            collision_loss = collision_loss * mask
            collision_loss = jnp.sum(collision_loss)

            # Penalize deviation from hover thrust in z, not absolute thrust
            # For xy, penalize absolute control; for z, penalize deviation from gravity compensation
            ctrl_loss = self.ctrl_weight * (jnp.square(ut[0]) + jnp.square(ut[1]) + jnp.square(ut[2] - self.g))
            # ctrl_loss = self.ctrl_weight * (jnp.square(ut[0]) + jnp.square(ut[1])) + self.ctrl_weight / 100.0 * jnp.square(ut[2] - self.g)
            return nav_loss + collision_loss + ctrl_loss
        
        def trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
            def single_step_loss(args):
                xt, ut, ref_xt, other_xts, mask = args
                return runtime_loss(xt, ut, ref_xt, other_xts, mask)
            loss_array = vmap(single_step_loss)((x_traj, u_traj, ref_x_traj, other_x_trajs, mask))
            return loss_array.sum() * self.dt

        def linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
            dldx = grad(runtime_loss, argnums=(0))
            dldu = grad(runtime_loss, argnums=(1))
            
            def grad_step(args):
                xt, ut, ref_xt, other_xts, mask = args
                return dldx(xt, ut, ref_xt, other_xts, mask), dldu(xt, ut, ref_xt, other_xts, mask)
            
            grads = vmap(grad_step)((x_traj, u_traj, ref_x_traj, other_x_trajs, mask))
            return grads[0], grads[1]  # a_traj, b_traj 
        
        self.runtime_loss = runtime_loss
        self.trajectory_loss = trajectory_loss
        self.linearize_loss = linearize_loss
        self.compiled_trajectory_loss = jit(trajectory_loss, device=self.device)
        self.compiled_linearize_loss = jit(linearize_loss, device=self.device)
        self.compiled_linearize_dyn = jit(self.linearize_dyn, device=self.device)
        self.compiled_solve = jit(self.solve, device=self.device)

    def create_loss_functions_no_mask(self):
        def runtime_loss(xt, ut, ref_xt, other_states):
            nav_loss = jnp.sum(jnp.square(xt[:3] - ref_xt[:3]))

            squared_distances = jnp.sum(jnp.square(xt[:3] - other_states[:, :3]), axis=1)
            collision_loss = self.collision_weight * jnp.exp(-self.collision_scale * squared_distances)
            collision_loss = jnp.sum(collision_loss)

            # Penalize deviation from hover thrust in z, not absolute thrust
            # For xy, penalize absolute control; for z, penalize deviation from gravity compensation
            ctrl_loss = self.ctrl_weight * (jnp.square(ut[0]) + jnp.square(ut[1]) + jnp.square(ut[2] - self.g))
            # ctrl_loss = self.ctrl_weight * (jnp.square(ut[0]) + jnp.square(ut[1])) + self.ctrl_weight / 100.0 * jnp.square(ut[2] - self.g)
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
