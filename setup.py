# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import find_packages, setup

setup(
    name="Pneusam",
    python_requires=">=3.9",
    install_requires=["monai==1.2.0", "pandas==2.1.1", "matplotlib==3.8.0", "scikit-image==0.21.0", "SimpleITK>=2.2.1",
                      "nibabel==5.1.0", "tqdm==4.66.1", "scipy==1.11.3", "ipympl==0.9.3", "opencv-python==4.8.1.78",
                      "jupyterlab==4.0.6", "ipywidgets==8.1.1", "flask==2.2.3", "flask_cors==3.0.10"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
