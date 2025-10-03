import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import jax
import jax.numpy as jnp
import zarr
from typing import Iterator, Tuple
import os


class MLPDataset:
    """
    TensorFlow dataloader for MLP trajectory data stored in Zarr format.
    Loads inputs, x0s, ref_trajs, and targets from separate Zarr arrays and outputs JAX numpy arrays.
    """
    
    def __init__(
        self, 
        inputs_path: str, 
        targets_path: str,
        x0s_path: str,
        ref_trajs_path: str,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        prefetch_size: int = 2
    ):
        """
        Initialize the MLP dataset loader.
        
        Args:
            inputs_path: Path to Zarr array containing input data
            targets_path: Path to Zarr array containing target data
            x0s_path: Path to Zarr array containing initial states
            ref_trajs_path: Path to Zarr array containing reference trajectories
            batch_size: Number of samples per batch
            shuffle_buffer: Size of shuffle buffer for randomization
            prefetch_size: Number of batches to prefetch
        """
        self.inputs_path = inputs_path
        self.targets_path = targets_path
        self.x0s_path = x0s_path
        self.ref_trajs_path = ref_trajs_path
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_size = prefetch_size
        
        # Validate that all Zarr files exist
        if not os.path.exists(inputs_path):
            raise FileNotFoundError(f"Inputs file not found: {inputs_path}")
        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"Targets file not found: {targets_path}")
        if not os.path.exists(x0s_path):
            raise FileNotFoundError(f"X0s file not found: {x0s_path}")
        if not os.path.exists(ref_trajs_path):
            raise FileNotFoundError(f"Ref trajs file not found: {ref_trajs_path}")
            
        # Load array metadata to get shapes
        self.inputs_array = zarr.open_array(inputs_path, mode='r')
        self.targets_array = zarr.open_array(targets_path, mode='r')
        self.x0s_array = zarr.open_array(x0s_path, mode='r')
        self.ref_trajs_array = zarr.open_array(ref_trajs_path, mode='r')
        
        # Validate that all arrays have the same number of samples
        self.num_samples = self.inputs_array.shape[0]
        if (self.targets_array.shape[0] != self.num_samples or 
            self.x0s_array.shape[0] != self.num_samples or 
            self.ref_trajs_array.shape[0] != self.num_samples):
            raise ValueError("All data arrays must have the same number of samples")
        
        self.input_dim = self.inputs_array.shape[1]
        self.target_shape = self.targets_array.shape[1:]
        self.x0s_dim = self.x0s_array.shape[1]
        self.ref_trajs_shape = self.ref_trajs_array.shape[1:]  # (30, 2) for 2D arrays
        
        print(f"Dataset loaded: {self.num_samples} samples")
        print(f"Input shape: {self.inputs_array.shape}")
        print(f"Target shape: {self.targets_array.shape} -> per sample: {self.target_shape}")
        print(f"X0s shape: {self.x0s_array.shape}")
        print(f"Ref trajs shape: {self.ref_trajs_array.shape} -> per sample: {self.ref_trajs_shape}")
    
    def _zarr_generator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generator that yields (inputs, x0s, ref_trajs, targets) tuples from Zarr arrays.
        All data comes from the same index i to ensure consistency.
        """
        for i in range(self.num_samples):
            # Load data as NumPy arrays from the same index
            input_data = np.asarray(self.inputs_array[i], dtype=np.float32)
            x0s_data = np.asarray(self.x0s_array[i], dtype=np.float32)
            ref_trajs_data = np.asarray(self.ref_trajs_array[i], dtype=np.float32)
            target_data = np.asarray(self.targets_array[i], dtype=np.float32)
            yield input_data, x0s_data, ref_trajs_data, target_data
    
    def create_tf_dataset(self) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from the Zarr data.
        
        Returns:
            tf.data.Dataset that yields (inputs, x0s, ref_trajs, targets) batches
        """
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            self._zarr_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.input_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=(self.x0s_dim,), dtype=tf.float32),
                tf.TensorSpec(shape=self.ref_trajs_shape, dtype=tf.float32),  # (30, 2)
                tf.TensorSpec(shape=self.target_shape, dtype=tf.float32)  # (4,)
            )
        )
        
        # Apply transformations
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.prefetch_size)
        
        return dataset
    
    def create_jax_iterator(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Create an iterator that yields JAX numpy arrays.
        
        Returns:
            Iterator yielding (inputs_batch, x0s_batch, ref_trajs_batch, targets_batch) as JAX arrays
        """
        tf_dataset = self.create_tf_dataset()
        
        for batch in tfds.as_numpy(tf_dataset):
            inputs_np, x0s_np, ref_trajs_np, targets_np = batch
            
            # Convert to JAX arrays and move to device
            inputs_jax = jax.device_put(jnp.asarray(inputs_np))
            x0s_jax = jax.device_put(jnp.asarray(x0s_np))
            ref_trajs_jax = jax.device_put(jnp.asarray(ref_trajs_np))
            targets_jax = jax.device_put(jnp.asarray(targets_np))
            
            yield inputs_jax, x0s_jax, ref_trajs_jax, targets_jax
    
    def get_batch_iterator(self, num_epochs: int = 1) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Get a batch iterator for training.
        
        Args:
            num_epochs: Number of epochs to iterate through
            
        Returns:
            Iterator yielding (inputs_batch, x0s_batch, ref_trajs_batch, targets_batch) as JAX arrays
        """
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            yield from self.create_jax_iterator()


def create_mlp_dataloader(
    inputs_path: str, 
    targets_path: str,
    x0s_path: str,
    ref_trajs_path: str,
    batch_size: int = 32,
    shuffle_buffer: int = 1000,
    prefetch_size: int = 2
) -> MLPDataset:
    """
    Convenience function to create an MLP dataset loader.
    
    Args:
        inputs_path: Path to inputs Zarr file
        targets_path: Path to targets Zarr file
        x0s_path: Path to x0s Zarr file
        ref_trajs_path: Path to ref_trajs Zarr file
        batch_size: Batch size for training
        shuffle_buffer: Shuffle buffer size
        prefetch_size: Prefetch size
        
    Returns:
        MLPDataset instance
    """
    return MLPDataset(
        inputs_path=inputs_path,
        targets_path=targets_path,
        x0s_path=x0s_path,
        ref_trajs_path=ref_trajs_path,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        prefetch_size=prefetch_size
    )


# Example usage
if __name__ == "__main__":
    # Create dataset loader
    dataset = create_mlp_dataloader(
        inputs_path="src/data/mlp_n_agents_10_test/inputs_test.zarr",
        targets_path="src/data/mlp_n_agents_10_test/targets_test.zarr",
        x0s_path="src/data/mlp_n_agents_10_test/x0s_test.zarr",
        ref_trajs_path="src/data/mlp_n_agents_10_test/ref_trajs_test.zarr",
        batch_size=16,
        shuffle_buffer=500
    )
    
    # Test the dataloader
    print("Testing dataloader...")
    batch_iter = dataset.create_jax_iterator()
    
    for i, (inputs, x0s, ref_trajs, targets) in enumerate(batch_iter):
        print(f"Batch {i}:")
        print(f"  Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
        print(f"  X0s shape: {x0s.shape}, dtype: {x0s.dtype}")
        print(f"  Ref trajs shape: {ref_trajs.shape}, dtype: {ref_trajs.dtype}")
        print(f"  Targets shape: {targets.shape}, dtype: {targets.dtype}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("Dataloader test completed successfully!")
