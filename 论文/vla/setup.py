"""
VLA Robot Learning Framework Setup Script
"""
from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "VLA Robot Learning Framework"

setup(
    name="vla-robot-learning",
    version="0.1.0",
    description="Minimal runnable system for robot learning with MuJoCo",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="VLA Team",
    author_email="vla@example.com",
    url="https://github.com/vla/vla-robot-learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "isort>=5.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "ros": [
            "rospy>=1.15.0",
            "rospkg>=1.4.0",
        ],
        "gpu": [
            "cupy-cuda11x>=10.0.0",
        ],
        "all": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "isort>=5.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "rospy>=1.15.0",
            "rospkg>=1.4.0",
            "cupy-cuda11x>=10.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    keywords="robotics, machine-learning, reinforcement-learning, mujoco, simulation",
    entry_points={
        "console_scripts": [
            "vla-train=vla.scripts.train:main",
            "vla-collect=vla.scripts.collect:main",
            "vla-evaluate=vla.scripts.evaluate:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/vla/vla-robot-learning/issues",
        "Source": "https://github.com/vla/vla-robot-learning",
        "Documentation": "https://vla-robot-learning.readthedocs.io/",
    },
)