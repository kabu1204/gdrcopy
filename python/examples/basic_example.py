#!/usr/bin/env python3
"""
Basic example of using the GDRCopy Python bindings.
"""

import os
import sys
import numpy as np
import ctypes

# Add the parent directory to the path so we can import the gdrcopy package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gdrcopy import GDRCopy, GDR_PIN_FLAG_DEFAULT, get_version

def main():
    """Main function."""
    try:
        # Print the GDRCopy version
        major, minor = get_version()
        print(f"GDRCopy version: {major}.{minor}")
        
        # Initialize CUDA
        try:
            import cupy as cp
            
            # Allocate GPU memory
            gpu_array = cp.arange(1000, dtype=np.float32)
            gpu_ptr = gpu_array.data.ptr
            
            print(f"Allocated GPU memory at address: 0x{gpu_ptr:x}")
            print(f"GPU array: {gpu_array[:10]}...")
            
            # Initialize GDRCopy
            gdr = GDRCopy()
            gdr.open()
            
            try:
                # Pin the GPU buffer
                handle = gdr.pin_buffer_v2(gpu_ptr, gpu_array.nbytes, GDR_PIN_FLAG_DEFAULT)
                
                # Get info about the pinned buffer
                info = handle.get_info()
                print(f"Pinned buffer info: {info}")
                
                # Map the GPU buffer to CPU accessible memory
                mapped_ptr = handle.map(gpu_array.nbytes)
                print(f"Mapped GPU memory to CPU address: 0x{mapped_ptr:x}")
                
                # Allocate host memory
                host_array = np.zeros_like(gpu_array.get(), dtype=np.float32)
                host_ptr = host_array.ctypes.data
                
                # Copy data from GPU to host using GDRCopy
                handle.copy_from_mapping(host_ptr, gpu_array.nbytes)
                
                print(f"Host array after copy from GPU: {host_array[:10]}...")
                
                # Modify host array
                host_array += 100
                print(f"Host array after modification: {host_array[:10]}...")
                
                # Copy data from host to GPU using GDRCopy
                handle.copy_to_mapping(host_ptr, gpu_array.nbytes)
                
                # Verify the changes
                result = cp.asnumpy(gpu_array)
                print(f"GPU array after copy from host: {result[:10]}...")
                
            finally:
                # Clean up
                gdr.close()
                
        except ImportError:
            print("CuPy not found. Please install CuPy to run this example.")
            print("You can install CuPy with: pip install cupy")
            return
            
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 