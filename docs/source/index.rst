==============================================================
pyMAISE: Michigan Artificial Intelligence Standard Environment
==============================================================

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. image:: https://github.com/aims-umich/pyMAISE/actions/workflows/CI.yml/badge.svg
   :target: https://github.com/aims-umich/pyMAISE/actions/workflows
.. image:: https://readthedocs.org/projects/pymaise/badge/?version=latest
   :target: https://pymaise.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/pyMAISE?color=teal
   :target: https://pypi.org/project/pyMAISE/

pyMAISE is an artificial intelligence (AI) and machine learning (ML) benchmarking library for nuclear reactor applications. It offers to streamline the building, tuning, comparison, and explainability of various ML models for user-provided data sets. Also, pyMAISE offers benchmarked data sets, written in Jupyter Notebooks, for AI/ML comparison. Current ML algorithm support includes

- linear regression,
- lasso regression,
- ridge regression,
- elastic net regression,
- logistic regression,
- decision tree regression and classification,
- extra trees regression and classification,
- adaboost regression and classification,
- support vector regression and classification,
- random forest regression and classification,
- k-nearest neighbors regression and classification,
- gaussian process regression and classification,
- gradient boosting regression and classification,
- sequential neural networks.

pyMAISE now includes support for ensemble methods, expanding its capabilities to enhance classical models. Currently, it offers the following options

- stacking,
- multi-output,

These models are built using `scikit-learn <https://scikit-learn.org/stable/index.html>`_ and `Keras <https://keras.io>`_ with explainability using `SHAP <https://shap.readthedocs.io/en/latest/index.html>`_ :cite:`scikit-learn, chollet2015keras, NIPS2017_7062`. pyMAISE supports the following neural network layers:

- dense,
- dropout,
- LSTM,
- GRU,
- 1D, 2D, and 3D convolutional,
- 1D, 2D, and 3D max pooling,
- flatten,
- and reshape.

Request further neural network layer support as an issue on the `pyMAISE repository <https://github.com/aims-umich/pyMAISE>`_. Refer to the sections below for more information, including installation, examples, and use. Use the :ref:`examples` as examples on pyMAISE functionality.

.. admonition:: Recommended publication for citing :cite:`MYERS2025105568`, 
   :class: tip

   .. code-block:: latex

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

--------
Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   user_guide
   dev_guide
   release_notes
   models
   pymaise_api
   benchmarks
   data_refs
   software_refs
   license
