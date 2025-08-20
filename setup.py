"""
Simple setup script
"""

from setuptools import setup, find_packages

setup(
    name="topoclassify",
    version="1.0.0",
    description="Topographic classification (Iwahashi & Pike, 2007) with Python",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "rasterio>=1.3.0",
        "scikit-image>=0.19.0",
        "matplotlib>=3.5.0"
    ],
    python_requires=">=3.7",
    author="Seniarwan",
    author_email="seniarwan@gmail.com",
    url="https://github.com/seniarwan/topo_classification"
)
