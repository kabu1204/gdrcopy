# GDRCopy Python Bindings

Python bindings for NVIDIA GPUDirect RDMA (GDRCopy) library.

## Overview

GDRCopy is a low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA technology. These Python bindings provide a convenient way to use GDRCopy from Python applications.

## Requirements

- Python 3.6 or later
- CFFI 1.0.0 or later
- NVIDIA CUDA Toolkit
- GDRCopy library (libgdrapi.so) installed on your system

## Installation

### From Source

```bash
# Install the GDRCopy library first
# Follow the instructions at https://github.com/NVIDIA/gdrcopy

# Then install the Python bindings
cd python
pip install .
```

## Usage

Here's a simple example of how to use the GDRCopy Python bindings with CUDA:

```python
import numpy as np
import ctypes
from gdrcopy import GDRCopy, GDR_PIN_FLAG_DEFAULT

# Initialize CUDA
import cupy as cp

# Allocate GPU memory
gpu_array = cp.arange(1000, dtype=np.float32)
gpu_ptr = gpu_array.data.ptr

# Initialize GDRCopy
gdr = GDRCopy()
gdr.open()

try:
    # Pin the GPU buffer
    handle = gdr.pin_buffer_v2(gpu_ptr, gpu_array.nbytes, GDR_PIN_FLAG_DEFAULT)
    
    # Map the GPU buffer to CPU accessible memory
    mapped_ptr = handle.map(gpu_array.nbytes)
    
    # Allocate host memory
    host_array = np.zeros_like(gpu_array, dtype=np.float32)
    host_ptr = host_array.ctypes.data
    
    # Copy data from GPU to host using GDRCopy
    handle.copy_from_mapping(host_ptr, gpu_array.nbytes)
    
    print("GPU array:", gpu_array)
    print("Host array:", host_array)
    
    # Modify host array
    host_array += 100
    
    # Copy data from host to GPU using GDRCopy
    handle.copy_to_mapping(host_ptr, gpu_array.nbytes)
    
    # Verify the changes
    result = cp.asnumpy(gpu_array)
    print("Modified GPU array:", result)
    
finally:
    # Clean up
    gdr.close()
```

## API Reference

### GDRCopy

The main class for interacting with the GDRCopy library.

- `open()`: Open a connection to the GDRCopy driver.
- `close()`: Close the connection to the GDRCopy driver.
- `pin_buffer(addr, size, p2p_token=0, va_space=0)`: Pin a GPU memory buffer for RDMA access.
- `pin_buffer_v2(addr, size, flags=GDR_PIN_FLAG_DEFAULT)`: Pin a GPU memory buffer for RDMA access (version 2).
- `get_attribute(attr)`: Get a GDRCopy attribute.

### GDRHandle

A handle to a pinned GPU memory buffer.

- `unpin()`: Unpin the GPU memory buffer.
- `get_info()`: Get information about the pinned GPU memory buffer.
- `map(size)`: Map the pinned GPU memory buffer to CPU accessible memory.
- `unmap()`: Unmap the GPU memory buffer.
- `copy_to_mapping(host_ptr, size)`: Copy data from host memory to the mapped GPU memory.
- `copy_from_mapping(host_ptr, size)`: Copy data from the mapped GPU memory to host memory.
- `get_callback_flag()`: Get the callback flag for this handle.

### Constants

- `GDR_MAPPING_TYPE_NONE`, `GDR_MAPPING_TYPE_WC`, `GDR_MAPPING_TYPE_CACHING`, `GDR_MAPPING_TYPE_DEVICE`: Mapping types.
- `GDR_PIN_FLAG_DEFAULT`, `GDR_PIN_FLAG_FORCE_PCIE`: Pinning flags.

### Functions

- `get_version()`: Get the runtime version of the GDRCopy library.
- `get_driver_version(gdr_handle)`: Get the driver version of the GDRCopy library.

## License

Same as GDRCopy - MIT License. 