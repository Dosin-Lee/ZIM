"""
Setup script for installing the `zim_anything` package.
"""

from setuptools import find_packages, setup

setup(
    name="zim_anything",
    version="0.1",
    install_requires=["onnx", "onnxruntime-gpu"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)

