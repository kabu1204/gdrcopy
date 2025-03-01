#!/usr/bin/env python3
"""
Advanced example of using the GDRCopy Python bindings with CUDA.
This example demonstrates how to use GDRCopy to efficiently transfer data
between the GPU and CPU.
"""

import os
import sys
import time
import numpy as np
import ctypes

# Add the parent directory to the path so we can import the gdrcopy package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gdrcopy import GDRCopy, GDR_PIN_FLAG_DEFAULT, GDR_PIN_FLAG_FORCE_PCIE

def benchmark_transfer(size_mb, use_force_pcie=False):
    """Benchmark data transfer between GPU and CPU using GDRCopy.
    
    Args:
        size_mb: Size of the data to transfer in MB.
        use_force_pcie: Whether to use the FORCE_PCIE flag.
        
    Returns:
        Tuple of (gpu_to_cpu_bw, cpu_to_gpu_bw) in GB/s.
    """
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not found. Please install CuPy to run this example.")
        print("You can install CuPy with: pip install cupy")
        return None, None
    
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Allocate GPU memory
    gpu_array = cp.ones(size_bytes // 4, dtype=np.float32)  # 4 bytes per float32
    gpu_ptr = gpu_array.data.ptr
    
    # Allocate host memory
    host_array = np.ones_like(gpu_array.get(), dtype=np.float32)
    host_ptr = host_array.ctypes.data
    
    # Initialize GDRCopy
    gdr = GDRCopy()
    gdr.open()
    
    try:
        # Pin the GPU buffer
        flags = GDR_PIN_FLAG_FORCE_PCIE if use_force_pcie else GDR_PIN_FLAG_DEFAULT
        handle = gdr.pin_buffer_v2(gpu_ptr, gpu_array.nbytes, flags)
        
        # Map the GPU buffer to CPU accessible memory
        mapped_ptr = handle.map(gpu_array.nbytes)
        
        # Benchmark GPU to CPU transfer
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            handle.copy_from_mapping(host_ptr, gpu_array.nbytes)
        end_time = time.time()
        gpu_to_cpu_time = (end_time - start_time) / iterations
        gpu_to_cpu_bw = size_mb / gpu_to_cpu_time / 1000  # GB/s
        
        # Benchmark CPU to GPU transfer
        start_time = time.time()
        for _ in range(iterations):
            handle.copy_to_mapping(host_ptr, gpu_array.nbytes)
        end_time = time.time()
        cpu_to_gpu_time = (end_time - start_time) / iterations
        cpu_to_gpu_bw = size_mb / cpu_to_gpu_time / 1000  # GB/s
        
        return gpu_to_cpu_bw, cpu_to_gpu_bw
    
    finally:
        # Clean up
        gdr.close()


def compare_with_cupy_transfer(size_mb):
    """Compare GDRCopy transfer with CuPy's built-in transfer.
    
    Args:
        size_mb: Size of the data to transfer in MB.
        
    Returns:
        Tuple of (gdrcopy_bw, cupy_bw) in GB/s.
    """
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not found. Please install CuPy to run this example.")
        print("You can install CuPy with: pip install cupy")
        return None, None
    
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Allocate GPU memory
    gpu_array = cp.ones(size_bytes // 4, dtype=np.float32)  # 4 bytes per float32
    gpu_ptr = gpu_array.data.ptr
    
    # Allocate host memory
    host_array = np.ones_like(gpu_array.get(), dtype=np.float32)
    
    # Benchmark CuPy transfer (GPU to CPU)
    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        host_array = cp.asnumpy(gpu_array)
    end_time = time.time()
    cupy_time = (end_time - start_time) / iterations
    cupy_bw = size_mb / cupy_time / 1000  # GB/s
    
    # Benchmark GDRCopy transfer
    gdr_bw, _ = benchmark_transfer(size_mb)
    
    return gdr_bw, cupy_bw


def main():
    """Main function."""
    print("GDRCopy CUDA Integration Example")
    print("================================")
    
    # Test different data sizes
    sizes = [1, 10, 100, 500]  # MB
    
    print("\nBenchmarking GDRCopy data transfer:")
    print("Size (MB) | GPU->CPU (GB/s) | CPU->GPU (GB/s)")
    print("-" * 50)
    
    for size in sizes:
        gpu_to_cpu_bw, cpu_to_gpu_bw = benchmark_transfer(size)
        if gpu_to_cpu_bw is not None:
            print(f"{size:9} | {gpu_to_cpu_bw:14.2f} | {cpu_to_gpu_bw:14.2f}")
    
    print("\nComparing GDRCopy with CuPy transfer (GPU->CPU):")
    print("Size (MB) | GDRCopy (GB/s) | CuPy (GB/s) | Speedup")
    print("-" * 60)
    
    for size in sizes:
        gdr_bw, cupy_bw = compare_with_cupy_transfer(size)
        if gdr_bw is not None:
            speedup = gdr_bw / cupy_bw if cupy_bw > 0 else float('inf')
            print(f"{size:9} | {gdr_bw:14.2f} | {cupy_bw:11.2f} | {speedup:7.2f}x")
    
    # Test with FORCE_PCIE flag
    print("\nTesting with FORCE_PCIE flag:")
    print("Size (MB) | GPU->CPU (GB/s) | CPU->GPU (GB/s)")
    print("-" * 50)
    
    for size in sizes:
        gpu_to_cpu_bw, cpu_to_gpu_bw = benchmark_transfer(size, use_force_pcie=True)
        if gpu_to_cpu_bw is not None:
            print(f"{size:9} | {gpu_to_cpu_bw:14.2f} | {cpu_to_gpu_bw:14.2f}")


if __name__ == "__main__":
    main() 