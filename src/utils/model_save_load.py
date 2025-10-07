import os
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


def _flatten_arrays(pytree): 
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    # Move off device -> host numpy for saving
    leaves = [jax.device_get(x) for x in leaves]
    return leaves, treedef


def save_model_npz(path: str, model) -> None:
    """Save only array leaves of an Equinox model to a .npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays, _static = eqx.partition(model, eqx.is_array)          # keep structure of arrays
    leaves, _ = _flatten_arrays(arrays)
    # Passing *args → np.savez names them arr_0, arr_1, ...
    np.savez(path, *leaves)


def load_model_npz(path: str, model_template):
    """
    Load array leaves from a .npz and combine with the non-array
    (static) parts from `model_template`. The template MUST have the
    same architecture/shape structure as the saved model.
    """
    arrays_template, static = eqx.partition(model_template, eqx.is_array)
    templ_leaves, treedef = jax.tree_util.tree_flatten(arrays_template)

    data = np.load(path)
    # np.savez with *args ⇒ keys are arr_0, arr_1, ...
    new_leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(len(templ_leaves))]
    new_arrays = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return eqx.combine(new_arrays, static)
