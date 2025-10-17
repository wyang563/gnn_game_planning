import jax
import jax.numpy as jnp
import jax.random
from jax import jit, vmap, grad
from jax import debug
import optax
from flax import linen as nn
from flax.training import train_state
from utils.utils import load_config, parse_arguments
from models.policies import *
from models.mlp import MLP
from data.mlp_dataset import create_mlp_dataloader
import os
from datetime import datetime
from lqrax import iLQR
import matplotlib.pyplot as plt
import numpy as np

# Configure JAX to use CPU if CUDA has issues
try:
    _ = jax.devices("cuda")
except:
    jax.config.update("jax_platform_name", "cpu")

class PointAgent(iLQR):
    """
    Point mass agent for trajectory optimization.
    
    State: [x, y, vx, vy] - position (x,y) and velocity (vx, vy)
    Control: [ax, ay] - acceleration in x and y directions
    
    Dynamics:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = ax
        dvy/dt = ay
    """
    def __init__(self, dt, x_dim, u_dim, Q, R, W):
        super().__init__(dt, x_dim, u_dim, Q, R)
        self.w1 = W[0]
        self.w2 = W[1]
        self.w3 = W[2]
        self.w4 = W[3]
    
    def dyn(self, x, u):
        """Dynamics function for point mass."""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])

    def runtime_loss(self, xt, ut, ref_xt, other_xts, mask):
        # calculate collision loss
        current_position = xt[:2]
        squared_distances = jnp.sum(jnp.square(current_position - other_xts), axis=1)
        collision_loss = self.w1 * jnp.exp(-self.w2 * squared_distances)
        # Apply mask to collision loss, not distance
        # alpha = 2.0
        # modified_mask = jax.nn.sigmoid(alpha * (mask - 0.5))
        collision_loss = collision_loss * mask 
        collision_loss = jnp.sum(collision_loss) 

        ctrl_loss: jnp.ndarray = self.w3 * jnp.sum(jnp.square(ut))
        nav_loss: jnp.ndarray = self.w4 * jnp.sum(jnp.square(xt[:2]-ref_xt[:2]))
        return nav_loss + collision_loss + ctrl_loss

    def loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
        runtime_loss_array = vmap(self.runtime_loss, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask)
        return runtime_loss_array.sum() * self.dt

    def linearize_loss(self, x_traj, u_traj, ref_x_traj, other_x_trajs, mask):
        dldx = grad(self.runtime_loss, argnums=(0))
        dldu = grad(self.runtime_loss, argnums=(1))
        a_traj = vmap(dldx, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask)  
        b_traj = vmap(dldu, in_axes=(0, 0, 0, 0, 0))(x_traj, u_traj, ref_x_traj, other_x_trajs, mask) 
        return a_traj, b_traj

class SimulatorTrain:
    def __init__(self, **kwargs):
        self.epochs = kwargs["epochs"]
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.model_type = kwargs["model_type"]
        self.n_agents = kwargs["n_agents"]
        self.mask_horizon = kwargs["mask_horizon"]
        self.step_size = kwargs.get("step_size", 0.002)
        self.dt = kwargs.get("dt", 0.1)
        self.state_dim = kwargs["state_dim"]
        self.batch_size = kwargs["batch_size"]
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.ego_agent_id = kwargs.get("ego_agent_id", 0)
        device_str = kwargs.get("device", "cpu")
        self.device = jax.devices(device_str)[0]
        
        # Cost matrices
        self.Q = jnp.diag(jnp.array(kwargs.get("Q", [1.0, 1.0, 0.1, 0.1])))
        self.R = jnp.diag(jnp.array(kwargs.get("R", [0.1, 0.1])))
        self.W = jnp.array(kwargs.get("W", [100.0, 5.0, 0.1, 1.0]))
        
        self.sigma_1 = kwargs["sigma_1"]
        self.sigma_1s = jnp.linspace(0, self.sigma_1, self.epochs)
        # print("sigma_1 will increase from 0 to", self.sigma_1, "over", self.epochs, "epochs")
        self.sigma_2 = kwargs["sigma_2"]
        self.optimization_iters = kwargs["optimization_iters"]

        # setup dataset
        self.dataloader = self._setup_dataloader(
            model_type=kwargs["model_type"],
            dataset_path=kwargs["dataset_dir"],
            batch_size=self.batch_size,
        )
        
        # # Setup batched agent functions for optimization
        self._setup_batched_functions()
        self.debug = kwargs["debug"]

    def _setup_batched_functions(self):
        """Setup batched jitted functions for better GPU utilization."""
        # Create a dummy agent to extract the methods we need to batch
        dummy_agent = PointAgent(dt=self.dt, x_dim=4, u_dim=2, Q=self.Q, R=self.R, W=self.W)

        # Define batched functions that work on arrays of agent data
        def batched_linearize_dyn(x0s, u_trajs):
            """Batched version of linearize_dyn for all agents."""
            def single_agent_linearize_dyn(x0, u_traj):
                return dummy_agent.linearize_dyn(x0, u_traj)
            return vmap(single_agent_linearize_dyn)(x0s, u_trajs)
        
        def batched_linearize_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
            """Batched version of linearize_loss for all agents."""
            def single_agent_linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask):
                return dummy_agent.linearize_loss(x_traj, u_traj, ref_traj, other_traj, mask)
            return vmap(single_agent_linearize_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
        
        def batched_solve(A_trajs, B_trajs, a_trajs, b_trajs):
            """Batched version of solve for all agents."""
            def single_agent_solve(A_traj, B_traj, a_traj, b_traj):
                return dummy_agent.solve(A_traj, B_traj, a_traj, b_traj)
            return vmap(single_agent_solve)(A_trajs, B_trajs, a_trajs, b_trajs)
        
        def batched_loss(x_trajs, u_trajs, ref_trajs, other_trajs, masks):
            """Batched version of loss for all agents."""
            def single_agent_loss(x_traj, u_traj, ref_traj, other_traj, mask):
                return dummy_agent.loss(x_traj, u_traj, ref_traj, other_traj, mask)
            return vmap(single_agent_loss)(x_trajs, u_trajs, ref_trajs, other_trajs, masks)
        
        # JIT compile the batched functions
        self.jit_batched_linearize_dyn = jit(batched_linearize_dyn, device=self.device)
        self.jit_batched_linearize_loss = jit(batched_linearize_loss, device=self.device)
        self.jit_batched_solve = jit(batched_solve, device=self.device)
        self.jit_batched_loss = jit(batched_loss, device=self.device)

    def _setup_dataloader(
        self,
        model_type: str,
        dataset_path: str,
        batch_size: int,
    ):
        if model_type == "mlp":
            return create_mlp_dataloader(
                inputs_path=os.path.join(dataset_path, "inputs.zarr"),
                targets_path=os.path.join(dataset_path, "targets.zarr"),
                x0s_path=os.path.join(dataset_path, "x0s.zarr"),
                ref_trajs_path=os.path.join(dataset_path, "ref_trajs.zarr"),
                shuffle_buffer=500,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")
    
    def binary_loss(self, masks):
        return jnp.mean(jnp.abs(0.5 - jnp.abs(0.5 - masks)))
    
    def mask_sparsity_loss(self, masks):
        return jnp.mean(masks)
    
    def similarity_loss(self, pred_traj, target_traj):
        pred_positions = pred_traj[:, :2]
        target_positions = target_traj[:, :2]
        position_diff = pred_positions - target_positions
        squared_distances = jnp.sum(jnp.square(position_diff), axis=-1)
        distances = jnp.sqrt(jnp.maximum(squared_distances, 1e-8))
        mean_distance = jnp.mean(distances)
        mean_distance = jnp.clip(mean_distance, 0.0, 1e3)
        return mean_distance
        
    def solve_masked_game_differentiable(self, sim_x0s, sim_ref_trajs, mask):
        # adjust masks such that they are n_agents long
        # only use masks for the ego agent, every other agent should consider all other agents
        masks = (~jnp.eye(self.n_agents, dtype=bool)).astype(jnp.float32)
        non_ego_indices = jnp.arange(1, self.n_agents)
        masks = masks.at[self.ego_agent_id, non_ego_indices].set(mask)

        horizon = sim_ref_trajs.shape[1]
        masks_for_step = jnp.tile(masks[:, None, :], (1, horizon, 1))

        # solve masked game over optimization iters
        u_trajs = jnp.zeros((self.n_agents, horizon, 2))
        for _ in range(self.optimization_iters):
            x_trajs, A_trajs, B_trajs = self.jit_batched_linearize_dyn(sim_x0s, u_trajs)
            all_x_pos = jnp.broadcast_to(x_trajs[None, :, :, :2], (self.n_agents, self.n_agents, horizon, 2))
            other_x_trajs = jnp.transpose(all_x_pos, (0, 2, 1, 3))
            a_trajs, b_trajs = self.jit_batched_linearize_loss(x_trajs, u_trajs, sim_ref_trajs, other_x_trajs, masks_for_step)
            v_trajs, _ = self.jit_batched_solve(A_trajs, B_trajs, a_trajs, b_trajs)
            u_trajs += self.step_size * v_trajs
        
        # get final x_trajs after optimization
        x_trajs, _, _ = self.jit_batched_linearize_dyn(sim_x0s, u_trajs)
        return x_trajs

    def compute_sim_loss_from_arrays(self, mask, sim_x0s, sim_ref_trajs, target_traj):
        state_trajectories = self.solve_masked_game_differentiable(
            sim_x0s,
            sim_ref_trajs,
            mask
        )
        # debug plot state trajectories with sim_ref_trajs and target_traj
        return self.similarity_loss(state_trajectories[self.ego_agent_id], target_traj)

    def batch_sim_loss(self, masks, x0s, ref_trajs, target_trajs):
        def per_sample(i):
            mask = masks[i]
            sim_x0s = x0s[i]
            sim_ref_trajs = ref_trajs[i]
            target_traj = target_trajs[i][self.ego_agent_id]
            return self.compute_sim_loss_from_arrays(mask, sim_x0s, sim_ref_trajs, target_traj)
        
        cur_batch_size = masks.shape[0]
        losses = jax.vmap(per_sample)(jnp.arange(cur_batch_size))
        return jnp.mean(losses)

    def train_step(self, state: train_state.TrainState, inputs, x0s, ref_trajs, targets):
        def loss_fn(params):
            predicted_masks = state.apply_fn({'params': params}, inputs)
            binary_loss_val = self.binary_loss(predicted_masks)
            mask_sparsity_loss_val = self.mask_sparsity_loss(predicted_masks)
            sim_loss = self.batch_sim_loss(predicted_masks, x0s, ref_trajs, targets)
            total_loss = binary_loss_val + self.sigma_1 * mask_sparsity_loss_val + self.sigma_2 * sim_loss
            return total_loss, (binary_loss_val, mask_sparsity_loss_val, sim_loss)

        (loss, loss_comps), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grad_norm = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), grads, initializer=0.0)
        grad_norm = jnp.sqrt(grad_norm)

        max_grad_norm = self.max_grad_norm
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / grad_norm
            grads = jax.tree.map(lambda g: g * scale, grads)
        
        # Apply gradients using the optimizer
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_comps 

    def create_train_state(self, model: nn.Module, optimizer, input_shape, rng):
        dummy_input = jnp.ones(input_shape)
        variables = model.init(rng, dummy_input)
        params = variables['params']
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        return state

    def plot_eval_trajs(self, state_trajectories, sim_input, sim_x0, sim_ref_trajs, target_traj, out_file_name):
        # convert arrays to numpy first
        state_trajectories = np.asarray(jax.device_get(state_trajectories))
        sim_input = np.asarray(jax.device_get(sim_input[0])) # have to unsqueeze the first dimension
        sim_x0 = np.asarray(jax.device_get(sim_x0))
        sim_ref_trajs = np.asarray(jax.device_get(sim_ref_trajs))
        target_traj = np.asarray(jax.device_get(target_traj))
        target_traj = target_traj[self.ego_agent_id][:, :2]

        plt.figure(figsize=(10, 10))
        plt.plot(state_trajectories[self.ego_agent_id, :, 0], state_trajectories[self.ego_agent_id, :, 1], linestyle="-.", label="Ego Actual Trajectory")
        plt.plot(sim_ref_trajs[self.ego_agent_id, :, 0], sim_ref_trajs[self.ego_agent_id, :, 1], linestyle="--", label="Ego Sim Ref Trajectory")
        plt.plot(target_traj[:, 0], target_traj[:, 1], linestyle=":", label="Ego Target Trajectory")
        plt.scatter(sim_x0[self.ego_agent_id, 0], sim_x0[self.ego_agent_id, 1], label="Ego Sim X0")
        plt.plot(sim_input[self.ego_agent_id, :, 0], sim_input[self.ego_agent_id, :, 1], linestyle="-", label="Ego Sim Input")
        plt.legend()
        plt.savefig(os.path.join(self.output_train_dir, "traj_plots", out_file_name))
        plt.close()

    def evaluate_model(self, state: train_state.TrainState, input, x0, ref_traj, target, epoch, iter):
        # Detach all inputs from gradient tracking and ensure single-sample shapes
        input = jax.lax.stop_gradient(input)
        x0 = jax.lax.stop_gradient(x0)
        ref_traj = jax.lax.stop_gradient(ref_traj)
        target = jax.lax.stop_gradient(target)

        predicted_mask = state.apply_fn({'params': state.params}, input)
        predicted_mask = jax.lax.stop_gradient(predicted_mask)
        # If model returns a leading batch dimension, squeeze/select a single sample
        if predicted_mask.ndim > 1:
            predicted_mask = predicted_mask[0]
        print(f"predicted_mask: {predicted_mask}", flush=True)
        state_trajectories = self.solve_masked_game_differentiable(x0, ref_traj, predicted_mask)
        self.plot_eval_trajs(state_trajectories, input, x0, ref_traj, target, out_file_name=f"epoch_{epoch}_iter_{iter}.png")

    def train(
            self, 
            model: MLP, 
            optimizer: optax.GradientTransformationExtraArgs, 
            rng: jnp.ndarray
        ): 
        
        output_train_dir = f"logs/train_{self.model_type}_{self.n_agents}_agents/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_train_dir = output_train_dir
        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(os.path.join(output_train_dir, "traj_plots"), exist_ok=True)
        os.makedirs(os.path.join(output_train_dir, "checkpoints"), exist_ok=True)

        input_shape = (self.batch_size, self.n_agents, self.mask_horizon, self.state_dim)
        state = self.create_train_state(model, optimizer, input_shape, rng)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.sigma_1 = self.sigma_1s[epoch]
            dataloader = self.dataloader.create_jax_iterator()
            epoch_loss = 0.0
            num_batches = 0
            batch_loss = 0.0

            for inputs, x0s, ref_trajs, targets in dataloader:
                state, loss, (binary_loss_val, mask_sparsity_loss_val, sim_loss_val) = self.train_step(state, inputs, x0s, ref_trajs, targets)

                # Accumulate loss for logging
                epoch_loss += float(loss)
                batch_loss += float(loss)
                num_batches += 1

                if num_batches % 20 == 0:
                    print(f"Running Avg Loss Batch {num_batches} - Loss: {batch_loss/20:.6f}", flush=True)
                    print(f"Binary Loss: {binary_loss_val:.6f}, Mask Sparsity Loss: {mask_sparsity_loss_val:.6f}, Sim Loss: {sim_loss_val:.6f}", flush=True)
                    batch_loss = 0.0

                    # evaluation (use a single sample and detach from gradients)
                    sample_input, sample_x0s, sample_ref_trajs, sample_targets = inputs[0], x0s[0], ref_trajs[0], targets[0]
                    sample_input = sample_input[None, ...]
                    self.evaluate_model(state, sample_input, sample_x0s, sample_ref_trajs, sample_targets, epoch, num_batches)

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.epochs} - Average Loss: {avg_loss:.6f}")

def setup_model(
    model_type: str,
    state_dim: int,
    n_agents: int,
    mask_horizon: int,
):
    if model_type == "mlp":
        model = MLP(hidden_dims=[128, 64, 32], n_agents=n_agents, mask_horizon=mask_horizon, state_dim=state_dim)
        return model
    # TODO: add GNN model later
    else:
        raise ValueError(f"Model type {model_type} not supported")

if __name__ == "__main__":
    DEBUG = False

    args = parse_arguments()
    
    # Load configuration from YAML file
    try:
        config = load_config(args.config)
        print(f"=== Training Receding Horizon Nash Game Simulator ===")
        print(f"Using config file: {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    model = setup_model(
        model_type=config["model_type"],
        state_dim=config["state_dim"],
        n_agents=config["n_agents"],
        mask_horizon=config["mask_horizon"],
    )

    optimizer = optax.adam(learning_rate=config["learning_rate"])
    # opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    trainer = SimulatorTrain(
        model_type=config["model_type"],
        batch_size=config["batch_size"],
        n_agents=config["n_agents"],
        mask_horizon=config["mask_horizon"],
        random_seed=config["random_seed"],
        epochs=config["epochs"],
        dataset_dir=config["dataset_dir"],
        sigma_1=config["sigma_1"],
        sigma_2=config["sigma_2"],
        optimization_iters=config["optimization_iters"],
        dt=config["dt"],
        state_dim=config["state_dim"],
        step_size=config["step_size"],
        Q=config["Q"],
        R=config["R"],
        W=config["W"],
        max_grad_norm=config["max_grad_norm"],
        debug=DEBUG,
    )

    rng = jax.random.PRNGKey(config["random_seed"])
    trainer.train(model=model, optimizer=optimizer, rng=rng)
    