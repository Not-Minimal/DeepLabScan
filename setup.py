"""
Setup configuration for DeepLabScan
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

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
        "ultralytics>=8.3.235",
        "torch>=2.9.1",
        "torchvision>=0.24.1",
        "opencv-python>=4.12.0.88",
        "numpy>=2.3.5",
        "matplotlib>=3.10.7",
        "pillow>=12.0.0",
        "roboflow>=1.2.11",
        "pyyaml>=6.0.3",
        "tqdm>=4.67.1",
        "scikit-learn>=1.7.2",
        "pandas>=2.3.3",
        "seaborn>=0.13.2",
        "python-dotenv>=1.2.1",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.2",
            "pytest-cov>=7.0.0",
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
