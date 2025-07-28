from setuptools import setup, find_packages

setup(
    name="meta-hrl",
    version="0.1.0",
    description="Meta-Learning for Hierarchical Skill Acquisition and Composition",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "gym>=0.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "full": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
            "pybullet>=3.2.0",
            "mujoco-py>=2.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)