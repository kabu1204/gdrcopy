"""
Python bindings for NVIDIA GPUDirect RDMA (GDRCopy) library.

This module provides Python bindings for the GDRCopy library, which enables
low-latency GPU memory copy using NVIDIA GPUDirect RDMA technology.
"""

from .gdrcopy import (
    GDRCopy,
    GDRError,
    GDRHandle,
    get_version,
    get_driver_version,
    GDR_MAPPING_TYPE_NONE,
    GDR_MAPPING_TYPE_WC,
    GDR_MAPPING_TYPE_CACHING,
    GDR_MAPPING_TYPE_DEVICE,
    GDR_PIN_FLAG_DEFAULT,
    GDR_PIN_FLAG_FORCE_PCIE
)

__all__ = [
    'GDRCopy',
    'GDRError',
    'GDRHandle',
    'get_version',
    'get_driver_version',
    'GDR_MAPPING_TYPE_NONE',
    'GDR_MAPPING_TYPE_WC',
    'GDR_MAPPING_TYPE_CACHING',
    'GDR_MAPPING_TYPE_DEVICE',
    'GDR_PIN_FLAG_DEFAULT',
    'GDR_PIN_FLAG_FORCE_PCIE'
] 