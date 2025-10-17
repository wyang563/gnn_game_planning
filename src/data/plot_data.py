import os
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from data.mlp_dataset import create_mlp_dataloader


def _get_dataset_paths() -> Tuple[str, str, str, str]:
    """
    Resolve absolute paths to the Zarr arrays located under the test dataset folder.
    """
    data_dir = Path(__file__).resolve().parent / "mlp_n_agents_10_test"
    inputs_path = str(data_dir / "inputs.zarr")
    targets_path = str(data_dir / "targets.zarr")
    x0s_path = str(data_dir / "x0s.zarr")
    ref_trajs_path = str(data_dir / "ref_trajs.zarr")
    return inputs_path, targets_path, x0s_path, ref_trajs_path


def _to_numpy(array: jnp.ndarray) -> np.ndarray:
    # Ensure we have a NumPy array on CPU for plotting
    return np.asarray(array)


def plot_single_sample(inputs: np.ndarray, ref_trajs: np.ndarray, targets: np.ndarray, output_path: str | None = None) -> None:
    """
    Plot, for each agent, the observed input trajectory (past), the reference
    trajectory to follow, and the target trajectory.

    Shapes (after selecting batch dimension):
      - inputs:   (N, T_past, 4)      -> use [:, :, 0:2] for (x, y)
      - ref_trajs:(N, T_future, 2)    -> already (x, y)
      - targets:  (N, T_future, 4)    -> use [:, :, 0:2] for (x, y)
    """
    num_agents = inputs.shape[0]

    # Extract XY components
    inputs_xy = inputs[:, :, 0:2]
    ref_xy = ref_trajs[:, :, 0:2]
    targets_xy = targets[:, :, 0:2]

    cmap = plt.cm.get_cmap("tab10", num_agents)
    plt.figure(figsize=(8, 8))

    for agent_idx in range(num_agents):
        color = cmap(agent_idx % 10)
        past = inputs_xy[agent_idx]  # (T_past, 2)
        ref = ref_xy[agent_idx]      # (T_future, 2)
        targ = targets_xy[agent_idx] # (T_future, 2)

        # Past inputs
        plt.plot(past[:, 0], past[:, 1], color=color, linewidth=2.0, label=(f"Agent {agent_idx} past" if agent_idx == 0 else None))
        plt.scatter(past[-1, 0], past[-1, 1], color=color, s=20)

        # Reference (dashed)
        plt.plot(ref[:, 0], ref[:, 1], color=color, linestyle="--", linewidth=1.5, label=("reference" if agent_idx == 0 else None))

        # Target (dotted)
        plt.plot(targ[:, 0], targ[:, 1], color=color, linestyle=":", linewidth=1.5, label=("target" if agent_idx == 0 else None))

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Per-agent trajectories: past inputs, reference, target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main() -> None:
    inputs_path, targets_path, x0s_path, ref_trajs_path = _get_dataset_paths()

    # Create a dataloader that yields one timestep (containing all agents) per batch
    dataset = create_mlp_dataloader(
        inputs_path=inputs_path,
        targets_path=targets_path,
        x0s_path=x0s_path,
        ref_trajs_path=ref_trajs_path,
        batch_size=1,
        shuffle_buffer=1,   # keep ordering deterministic here
        prefetch_size=1,
    )

    # Grab a single batch (shape: (1, N, ...))
    iterator = dataset.create_jax_iterator()
    try:
        inputs_b, _x0s_b, ref_b, targets_b = next(iter(iterator))
    except StopIteration:
        raise RuntimeError("Dataset appears to be empty; no batches to plot.")

    # Squeeze batch dimension
    inputs_np = _to_numpy(inputs_b)[0]
    ref_np = _to_numpy(ref_b)[0]
    targets_np = _to_numpy(targets_b)[0]

    # Save to project-level plots directory
    project_root = Path(__file__).resolve().parents[2]
    output_path = str(project_root / "plots" / "agent_trajectories.png")
    plot_single_sample(inputs_np, ref_np, targets_np, output_path=output_path)


if __name__ == "__main__":
    main()


