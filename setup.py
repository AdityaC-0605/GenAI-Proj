"""Setup script for Cross-Lingual QA System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cross-lingual-qa",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Cross-Lingual Question Answering System with mBERT and mT5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/cross-lingual-qa",
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "data"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clqa-train-zero-shot=scripts.train_zero_shot:main",
            "clqa-train-few-shot=scripts.train_few_shot:main",
            "clqa-evaluate=scripts.evaluate:main",
            "clqa-compare=scripts.compare_models:main",
            "clqa-serve=src.api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)
