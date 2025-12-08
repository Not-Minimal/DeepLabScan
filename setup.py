"""
Setup configuration for DeepLabScan
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="deeplabscan",
    version="1.0.0",
    author="DeepLabScan Team",
    description="Proyecto semestral de detección de objetos usando YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Not-Minimal/DeepLabScan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "roboflow>=1.1.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deeplabscan-train=scripts.train:main",
            "deeplabscan-evaluate=scripts.evaluate:main",
            "deeplabscan-predict=scripts.predict:main",
            "deeplabscan-download=scripts.download_data:main",
        ],
    },
)
