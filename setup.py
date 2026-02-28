"""
Setup script for TopoAdam optimizer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists()
    else "TopoAdam: Topology-Guided Adaptive Optimizer for PyTorch"
)

setup_args = dict(
    name="topoadamw",
    version="0.1.0",
    author="Kelvin Peng",
    author_email="kelvinpeng2004@outlook.com",
    description="Topology-guided adaptive optimizer for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SVAH-X/topoadamw",
    packages=["topoadamw"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "tda": ["gudhi>=3.5.0"],
        "dev": ["pytest", "matplotlib", "seaborn"],
        "benchmark": ["torchvision", "matplotlib", "seaborn"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

setup(**setup_args)
