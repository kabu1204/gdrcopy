#!/usr/bin/env python3
"""
Setup script for the gdrcopy Python bindings.
"""

from setuptools import setup, find_packages

setup(
    name="gdrcopy",
    version="2.5",  # Match the version of the GDRCopy library
    description="Python bindings for NVIDIA GPUDirect RDMA (GDRCopy) library",
    author="NVIDIA",
    author_email="",
    url="https://github.com/NVIDIA/gdrcopy",
    packages=find_packages(),
    install_requires=[
        "cffi>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.6",
) 