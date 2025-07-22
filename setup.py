"""
Setup script for PhotoSight
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="photosight",
    version="0.1.0",
    author="Sam",
    description="Intelligent RAW photo processing pipeline with scene-aware analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samscarrow/photosight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "rawpy>=0.18.1",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "PyExifTool>=0.5.5",
        "click>=8.1.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "color": [
            "colorlog>=6.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photosight=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "photosight": ["config.yaml"],
    },
)