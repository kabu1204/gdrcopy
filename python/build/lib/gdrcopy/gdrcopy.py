"""
Python bindings for NVIDIA GPUDirect RDMA (GDRCopy) library.
"""

import os
import sys
import ctypes
from typing import Tuple, Optional, Dict, Any

import cffi

# Define constants from gdrapi.h
GDR_MAPPING_TYPE_NONE = 0
GDR_MAPPING_TYPE_WC = 1
GDR_MAPPING_TYPE_CACHING = 2
GDR_MAPPING_TYPE_DEVICE = 3

GDR_PIN_FLAG_DEFAULT = 0
GDR_PIN_FLAG_FORCE_PCIE = 1

GDR_ATTR_USE_PERSISTENT_MAPPING = 1
GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2

# Define the C declarations for CFFI
_C_DECLARATIONS = """
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
    unsigned long h;
} gdr_mh_t;

typedef enum gdr_mapping_type {
    GDR_MAPPING_TYPE_NONE = 0,
    GDR_MAPPING_TYPE_WC = 1,
    GDR_MAPPING_TYPE_CACHING = 2,
    GDR_MAPPING_TYPE_DEVICE = 3
} gdr_mapping_type_t;

typedef struct gdr_info_v2 {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
    gdr_mapping_type_t mapping_type;
} gdr_info_v2_t;

typedef enum gdr_pin_flags {
    GDR_PIN_FLAG_DEFAULT = 0,
    GDR_PIN_FLAG_FORCE_PCIE = 1
} gdr_pin_flags_t;

typedef enum gdr_attr {
    GDR_ATTR_USE_PERSISTENT_MAPPING = 1,
    GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2,
    GDR_ATTR_MAX
} gdr_attr_t;

gdr_t gdr_open(void);
int gdr_close(gdr_t g);
int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle);
int gdr_pin_buffer_v2(gdr_t g, unsigned long addr, size_t size, uint32_t flags, gdr_mh_t *handle);
int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag);
int gdr_get_info_v2(gdr_t g, gdr_mh_t handle, gdr_info_v2_t *info);
int gdr_map(gdr_t g, gdr_mh_t handle, void **va, size_t size);
int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size);
int gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
int gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
void gdr_runtime_get_version(int *major, int *minor);
int gdr_driver_get_version(gdr_t g, int *major, int *minor);
int gdr_get_attribute(gdr_t g, gdr_attr_t attr, int *v);
"""

# Initialize CFFI
ffi = cffi.FFI()
ffi.cdef(_C_DECLARATIONS)

# Try to load the library
try:
    # First check if user specified a custom location via environment variable
    lib_path = os.environ.get("GDRCOPY_LIBRARY_PATH")
    if lib_path:
        _lib = ffi.dlopen(os.path.join(lib_path, "libgdrapi.so"))
    else:
        # Try to find the library in standard locations
        _lib = ffi.dlopen("libgdrapi.so")
except OSError:
    # If not found, try to find it in the parent directory (assuming we're in the Python package)
    try:
        _lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "src", "libgdrapi.so"))
    except OSError as e:
        raise ImportError(f"Could not load GDRCopy library: {e}. Make sure libgdrapi.so is installed or set GDRCOPY_LIBRARY_PATH environment variable.") from e


class GDRError(Exception):
    """Exception raised for GDRCopy errors."""
    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or f"GDRCopy error code: {code}"
        super().__init__(self.message)


def _check_error(ret: int, msg: str = ""):
    """Check if a GDRCopy function returned an error and raise an exception if so."""
    if ret != 0:
        raise GDRError(ret, f"{msg} (error code: {ret})")


def get_version() -> Tuple[int, int]:
    """Get the runtime version of the GDRCopy library.
    
    Returns:
        Tuple[int, int]: A tuple containing the major and minor version numbers.
    """
    major = ffi.new("int *")
    minor = ffi.new("int *")
    _lib.gdr_runtime_get_version(major, minor)
    return major[0], minor[0]


def get_driver_version(gdr_handle) -> Tuple[int, int]:
    """Get the driver version of the GDRCopy library.
    
    Args:
        gdr_handle: A GDRCopy handle obtained from GDRCopy.open().
        
    Returns:
        Tuple[int, int]: A tuple containing the major and minor version numbers.
    """
    major = ffi.new("int *")
    minor = ffi.new("int *")
    ret = _lib.gdr_driver_get_version(gdr_handle._handle, major, minor)
    _check_error(ret, "Failed to get driver version")
    return major[0], minor[0]


class GDRHandle:
    """A handle to a pinned GPU memory buffer."""
    
    def __init__(self, gdr, handle):
        """Initialize a GDRHandle.
        
        Args:
            gdr: The GDRCopy instance that created this handle.
            handle: The raw gdr_mh_t handle.
        """
        self._gdr = gdr
        self._handle = handle
        self._mapped_ptr = None
        self._mapped_size = 0
        self._info = None
    
    def __del__(self):
        """Clean up resources when the handle is garbage collected."""
        self.unmap()
        self.unpin()
    
    def unpin(self) -> None:
        """Unpin the GPU memory buffer."""
        if hasattr(self, '_handle') and self._handle:
            ret = _lib.gdr_unpin_buffer(self._gdr._handle, self._handle)
            self._handle = None
            _check_error(ret, "Failed to unpin buffer")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the pinned GPU memory buffer.
        
        Returns:
            Dict[str, Any]: A dictionary containing information about the pinned buffer.
        """
        if not self._info:
            info = ffi.new("gdr_info_v2_t *")
            ret = _lib.gdr_get_info_v2(self._gdr._handle, self._handle, info)
            _check_error(ret, "Failed to get info")
            
            self._info = {
                'va': info.va,
                'mapped_size': info.mapped_size,
                'page_size': info.page_size,
                'mapped': bool(info.mapped),
                'wc_mapping': bool(info.wc_mapping),
                'mapping_type': info.mapping_type
            }
        
        return self._info
    
    def map(self, size: int) -> int:
        """Map the pinned GPU memory buffer to CPU accessible memory.
        
        Args:
            size: The size of the buffer to map.
            
        Returns:
            int: The address of the mapped memory.
        """
        if self._mapped_ptr:
            raise RuntimeError("Buffer is already mapped")
        
        ptr = ffi.new("void **")
        ret = _lib.gdr_map(self._gdr._handle, self._handle, ptr, size)
        _check_error(ret, "Failed to map buffer")
        
        self._mapped_ptr = ptr[0]
        self._mapped_size = size
        
        return int(ffi.cast("uintptr_t", self._mapped_ptr))
    
    def unmap(self) -> None:
        """Unmap the GPU memory buffer."""
        if hasattr(self, '_mapped_ptr') and self._mapped_ptr and hasattr(self, '_mapped_size'):
            ret = _lib.gdr_unmap(self._gdr._handle, self._handle, self._mapped_ptr, self._mapped_size)
            self._mapped_ptr = None
            self._mapped_size = 0
            _check_error(ret, "Failed to unmap buffer")
    
    def copy_to_mapping(self, host_ptr: int, size: int) -> None:
        """Copy data from host memory to the mapped GPU memory.
        
        Args:
            host_ptr: The address of the host memory to copy from.
            size: The size of the data to copy.
        """
        if not self._mapped_ptr:
            raise RuntimeError("Buffer is not mapped")
        
        h_ptr = ffi.cast("void *", host_ptr)
        ret = _lib.gdr_copy_to_mapping(self._handle, self._mapped_ptr, h_ptr, size)
        _check_error(ret, "Failed to copy to mapping")
    
    def copy_from_mapping(self, host_ptr: int, size: int) -> None:
        """Copy data from the mapped GPU memory to host memory.
        
        Args:
            host_ptr: The address of the host memory to copy to.
            size: The size of the data to copy.
        """
        if not self._mapped_ptr:
            raise RuntimeError("Buffer is not mapped")
        
        h_ptr = ffi.cast("void *", host_ptr)
        ret = _lib.gdr_copy_from_mapping(self._handle, h_ptr, self._mapped_ptr, size)
        _check_error(ret, "Failed to copy from mapping")
    
    def get_callback_flag(self) -> bool:
        """Get the callback flag for this handle.
        
        Returns:
            bool: True if the callback flag is set, False otherwise.
        """
        flag = ffi.new("int *")
        ret = _lib.gdr_get_callback_flag(self._gdr._handle, self._handle, flag)
        _check_error(ret, "Failed to get callback flag")
        return bool(flag[0])


class GDRCopy:
    """Main class for interacting with the GDRCopy library."""
    
    def __init__(self):
        """Initialize a GDRCopy instance."""
        self._handle = None
    
    def __del__(self):
        """Clean up resources when the instance is garbage collected."""
        self.close()
    
    def open(self) -> None:
        """Open a connection to the GDRCopy driver."""
        if self._handle:
            raise RuntimeError("GDRCopy is already open")
        
        self._handle = _lib.gdr_open()
        if not self._handle:
            raise GDRError(-1, "Failed to open GDRCopy")
    
    def close(self) -> None:
        """Close the connection to the GDRCopy driver."""
        if hasattr(self, '_handle') and self._handle:
            ret = _lib.gdr_close(self._handle)
            self._handle = None
            _check_error(ret, "Failed to close GDRCopy")
    
    def pin_buffer(self, addr: int, size: int, p2p_token: int = 0, va_space: int = 0) -> GDRHandle:
        """Pin a GPU memory buffer for RDMA access.
        
        Args:
            addr: The GPU memory address to pin.
            size: The size of the buffer to pin.
            p2p_token: The peer-to-peer token (deprecated).
            va_space: The virtual address space (deprecated).
            
        Returns:
            GDRHandle: A handle to the pinned buffer.
        """
        if not self._handle:
            raise RuntimeError("GDRCopy is not open")
        
        handle = ffi.new("gdr_mh_t *")
        ret = _lib.gdr_pin_buffer(self._handle, addr, size, p2p_token, va_space, handle)
        _check_error(ret, "Failed to pin buffer")
        
        return GDRHandle(self, handle[0])
    
    def pin_buffer_v2(self, addr: int, size: int, flags: int = GDR_PIN_FLAG_DEFAULT) -> GDRHandle:
        """Pin a GPU memory buffer for RDMA access (version 2).
        
        Args:
            addr: The GPU memory address to pin.
            size: The size of the buffer to pin.
            flags: Flags for pinning the buffer.
            
        Returns:
            GDRHandle: A handle to the pinned buffer.
        """
        if not self._handle:
            raise RuntimeError("GDRCopy is not open")
        
        handle = ffi.new("gdr_mh_t *")
        ret = _lib.gdr_pin_buffer_v2(self._handle, addr, size, flags, handle)
        _check_error(ret, "Failed to pin buffer (v2)")
        
        return GDRHandle(self, handle[0])
    
    def get_attribute(self, attr: int) -> int:
        """Get a GDRCopy attribute.
        
        Args:
            attr: The attribute to get.
            
        Returns:
            int: The value of the attribute.
        """
        if not self._handle:
            raise RuntimeError("GDRCopy is not open")
        
        value = ffi.new("int *")
        ret = _lib.gdr_get_attribute(self._handle, attr, value)
        _check_error(ret, f"Failed to get attribute {attr}")
        
        return value[0] 