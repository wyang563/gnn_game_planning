#!/usr/bin/env python3
"""
Test script to verify the modified save_mlp_dataset function works with 3D data.
"""

import numpy as np
import zarr
import os
import tempfile
import sys

# Add the src directory to the path so we can import the Simulator
sys.path.append('/home/alex/gnn_game_planning/src')

from sim_solver import Simulator

def test_save_mlp_dataset_3d():
    """Test the save_mlp_dataset function with 3D data."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_3d_data.zarr")
        
        # Create a dummy simulator instance to access the method
        # We'll use minimal parameters since we only need the save_mlp_dataset method
        simulator = Simulator(
            n_agents=2,
            Q=np.eye(4),
            R=np.eye(2),
            W=np.array([1.0, 1.0]),
            time_steps=10,
            horizon=5,
            mask_horizon=3,
            dt=0.1,
            init_arena_range=(-2, 2),
            device="cpu"
        )
        
        # Test 1: Create new 3D array
        print("Test 1: Creating new 3D array...")
        data_3d_1 = np.random.rand(3, 4, 5).astype(np.float32)
        simulator.save_mlp_dataset(data_3d_1, test_file)
        
        # Verify the array was created correctly
        arr = zarr.open_array(test_file, mode='r')
        print(f"Created array shape: {arr.shape}")
        print(f"Expected shape: (3, 4, 5)")
        assert arr.shape == (3, 4, 5), f"Expected shape (3, 4, 5), got {arr.shape}"
        assert np.allclose(arr[:], data_3d_1), "Data mismatch in created array"
        print("✓ Test 1 passed: 3D array created correctly")
        
        # Test 2: Append more 3D data
        print("\nTest 2: Appending more 3D data...")
        data_3d_2 = np.random.rand(2, 4, 5).astype(np.float32)
        simulator.save_mlp_dataset(data_3d_2, test_file)
        
        # Verify the data was appended correctly
        arr = zarr.open_array(test_file, mode='r')
        print(f"After append array shape: {arr.shape}")
        print(f"Expected shape: (5, 4, 5)")
        assert arr.shape == (5, 4, 5), f"Expected shape (5, 4, 5), got {arr.shape}"
        assert np.allclose(arr[:3], data_3d_1), "First batch data mismatch"
        assert np.allclose(arr[3:], data_3d_2), "Second batch data mismatch"
        print("✓ Test 2 passed: 3D data appended correctly")
        
        # Test 3: Test with 2D data (backward compatibility)
        print("\nTest 3: Testing backward compatibility with 2D data...")
        test_file_2d = os.path.join(temp_dir, "test_2d_data.zarr")
        data_2d_1 = np.random.rand(3, 10).astype(np.float32)
        simulator.save_mlp_dataset(data_2d_1, test_file_2d)
        
        arr_2d = zarr.open_array(test_file_2d, mode='r')
        print(f"2D array shape: {arr_2d.shape}")
        print(f"Expected shape: (3, 10)")
        assert arr_2d.shape == (3, 10), f"Expected shape (3, 10), got {arr_2d.shape}"
        assert np.allclose(arr_2d[:], data_2d_1), "2D data mismatch"
        print("✓ Test 3 passed: 2D data still works")
        
        # Test 4: Test dimension mismatch error
        print("\nTest 4: Testing dimension mismatch error...")
        try:
            data_2d_wrong = np.random.rand(2, 10).astype(np.float32)
            simulator.save_mlp_dataset(data_2d_wrong, test_file)  # This should fail
            assert False, "Expected ValueError for dimension mismatch"
        except ValueError as e:
            print(f"✓ Test 4 passed: Correctly caught dimension mismatch: {e}")
        
        # Test 5: Test feature dimension mismatch error
        print("\nTest 5: Testing feature dimension mismatch error...")
        try:
            data_3d_wrong = np.random.rand(2, 3, 6).astype(np.float32)  # Wrong last dimension
            simulator.save_mlp_dataset(data_3d_wrong, test_file)  # This should fail
            assert False, "Expected ValueError for feature dimension mismatch"
        except ValueError as e:
            print(f"✓ Test 5 passed: Correctly caught feature dimension mismatch: {e}")
        
        print("\n🎉 All tests passed! The modified save_mlp_dataset function works correctly with 3D data.")

if __name__ == "__main__":
    test_save_mlp_dataset_3d()

