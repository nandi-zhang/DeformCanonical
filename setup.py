from setuptools import setup, find_packages

setup(
    name="canonical_ar",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "wandb>=0.16.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "trimesh>=4.0.0",
        "plyfile>=0.7.4",
    ],
    extras_require={
        # Install with: pip install -e ".[full]"
        "full": [
            "open3d>=0.17.0",   # better normal estimation; falls back gracefully if missing
        ],
    },
    # pytorch3d has no PyPI wheel — not required for current code.
    # Install torch first from https://pytorch.org/get-started/locally/
    # before running pip install -e .
)
