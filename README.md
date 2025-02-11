# pyMAISE: Michigan Artificial Intelligence Standard Environment

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests Status](https://github.com/aims-umich/pyMAISE/actions/workflows/CI.yml/badge.svg)](https://github.com/aims-umich/pyMAISE/actions/workflows)
[![Documentation Status](https://readthedocs.org/projects/pymaise/badge/?version=latest)](https://pymaise.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pyMAISE?color=teal)](https://pypi.org/project/pyMAISE/)

pyMAISE is an artificial intelligence (AI) and machine learning (ML) benchmarking library for nuclear reactor applications. It offers to streamline the building, tuning, comparison, and explainability of various ML models for user-provided data sets. Also, pyMAISE offers benchmarked data sets, written in Jupyter Notebooks, for AI/ML comparison. Current ML algorithm support includes

- linear regression,
- lasso regression,
- logistic regression,
- decision tree regression and classification,
- support vector regression and classification,
- random forest regression and classification,
- k-nearest neighbors regression and classification,
- sequential neural networks.

These models are built using [scikit-learn](https://scikit-learn.org/stable/index.html) and [Keras](https://keras.io) with explainability using [SHAP](https://shap.readthedocs.io/en/latest/index.html). pyMAISE supports the following neural network layers:

- dense,
- dropout,
- LSTM,
- GRU,
- 1D, 2D, and 3D convolutional,
- 1D, 2D, and 3D max pooling,
- flatten,
- and reshape.

Please use the following to reference pyMAISE:
```latex
@article{MYERS2025105568,
    title = {pyMAISE: A Python platform for automatic machine learning
        and accelerated development for nuclear power applications},
    journal = {Progress in Nuclear Energy},
    volume = {180},
    pages = {105568},
    year = {2025},
    issn = {0149-1970},
    doi = {https://doi.org/10.1016/j.pnucene.2024.105568},
    url = {
        https://www.sciencedirect.com/science/article/pii/S0149197024005183
    },
    author = {Patrick A. Myers and Nataly Panczyk and Shashank Chidige
        and Connor Craig and Jacob Cooper and Veda Joynt and Majdi
        I. Radaideh},
}
```

## Installation and Documentation

Refer to the [installation guide](https://pymaise.readthedocs.io/en/latest/installation.html) and [documentation](https://pymaise.readthedocs.io/en/latest/index.html) for help.

## Benchmark Jupyter Notebooks

You can find the pyMAISE benchmarks [here](https://pymaise.readthedocs.io/en/latest/benchmarks.html) or below.

- [MIT Reactor](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/mit_reactor.ipynb)
- [Reactor Physics](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/reactor_physics.ipynb)
- [Fuel Performance](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/fuel_performance.ipynb)
- [Heat Conduction](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/heat_conduction.ipynb)
- [BWR Micro Core](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/bwr.ipynb)
- [HTGR Micro-Core Quadrant Power](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/HTGR_microreactor.ipynb)
- [NEACRP C1 Rod Ejection Accident](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/rod_ejection.ipynb)
- [Critical Heat Flux (CHF) Prediction](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/chf.ipynb)
- [Binary Anomaly Detection](https://nbviewer.org/github/aims-umich/pyMAISE/blob/develop/docs/source/benchmarks/anomaly.ipynb)
