import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)

x = jnp.ones((5, 4))
y = jax.random.uniform(key, shape=(5,4), minval=0.0, maxval=1.0, dtype=jnp.float32)
print(x)
print(y)
diff = x - y
print(diff)
diff2 = jnp.square(diff)
print(diff2)
l2 = jnp.sqrt(jnp.sum(diff2, axis=1))
print(l2)
