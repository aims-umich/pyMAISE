import sys
import warnings

from setuptools import find_packages, setup

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

if sys.version_info < (3, 11) or sys.version_info > (3, 13):
    sys.exit("pyMAISE only supports python>=3.11 and python<=3.13")

# Get version from pyMAISE/__init__.py (always last line)
with open("pyMAISE/__init__.py") as f:
    version = f.readlines()[-1].split()[-1][1:-1]

setup(
    name="pyMAISE",
    version=version,
    packages=find_packages(include=["pyMAISE", "pyMAISE.*"]),
    install_requires=[
        "pandas",
        "numpy>=1.23,<2.0",
        "scikit-learn",
        "scikit-optimize==0.9.0",
        "torch",
        "captum>=0.7.0",
        "optuna",
        "skorch",
        "xarray>=2024.10.0",
        "matplotlib",
        "tqdm",
        "graphviz",
        "torchview",
        "pooch",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pre-commit",
            "flake8",
            "black",
            "docformatter",
            "sphinx",
            "sphinx_rtd_theme",
            "jupyter",
            "sphinxcontrib.bibtex",
            "chardet",
            "nbsphinx",
            "opencv-python",
            "scipy>=1.11,<1.15",
        ],
        "benchmarks": ["jupyter", "opencv-python", "scipy>=1.11,<1.15"],
    },
    description="Michigan Artificial Intelligance Standard Environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Patrick Myers",
    author_email="myerspat@umich.edu",
    project_urls={
        "Documentation": "https://pymaise.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/aims-umich/pyMAISE",
    },
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
    ],
)
