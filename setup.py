#!/usr/bin/env python
"""Lightweight install: grasp/centroid helpers only (no learned segmenter)."""

from setuptools import setup

setup(
    name="bin-picking-grasp-utils",
    version="0.1.0",
    description="Centroid and suction grasp utilities from instance masks",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19",
        "opencv-python>=4.5",
    ],
    extras_require={
        "viz": ["matplotlib>=3.3"],
        "dev": ["scikit-image>=0.18"],
    },
    py_modules=[],
)
