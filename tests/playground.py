import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(42)
batch_size = 8
n_agents = 10
t_horizon = 10
state_dim = 4
values = np.random.randn(batch_size, n_agents, t_horizon, state_dim)
x = jnp.array(values)

positions = x[:, :, :, :2]
velocities = x[:, :, :, 2:]
# note this assumes ego agent ind is 0
batch_last_ego_position = positions[:, 0, -1, :]
positions = positions - batch_last_ego_position[:, None, None, :]

x = jnp.concatenate([positions, velocities], axis=-1)
print("concatenated x")
print(x)
# Reshape input to (batch_size, T_observation, N_agents, state_dim)
batch_size = x.shape[0]
x = x.reshape(batch_size, n_agents * state_dim * t_horizon)
print("output x")
print(x)


