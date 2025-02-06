==============================================================
pyMAISE: Michigan Artificial Intelligence Standard Environment
==============================================================

pyMAISE is an artificial intelligence (AI) and machine learning (ML) benchmarking library for nuclear reactor applications. It offers to streamline the building, tuning, and comparison of various ML models for user-provided data sets. Also, pyMAISE offers benchmarked data sets, written in Jupyter Notebooks, for AI/ML comparison. Current ML algorithm support includes

- linear regression,
- lasso regression,
- logistic regression,
- decision tree regression and classification,
- support vector regression and classification,
- random forest regression and classification,
- k-nearest neighbors regression and classification,
- sequential neural networks.

These models are built using `scikit-learn <https://scikit-learn.org/stable/index.html>`_ and `Keras <https://keras.io>`_ :cite:`scikit-learn, chollet2015keras`. pyMAISE supports the following neural network layers:

- dense,
- dropout,
- LSTM,
- GRU,
- 1D, 2D, and 3D convolutional,
- 1D, 2D, and 3D max pooling,
- flatten,
- and reshape.

Additionally, pyMAISE supports basic explainability analysis via SHAP for all the ML algorithms listed above. Current SHAP support includes

- DeepLIFT,
- Integrated Gradients,
- Kernel SHAP,
- and Exact SHAP.

Request further neural network layer support as an issue on the `pyMAISE repository <https://github.com/myerspat/pyMAISE>`_. Refer to the sections below for more information, including installation, examples, and use. Use the :ref:`examples` as examples on pyMAISE functionality.

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
   license

.. _data_refs:

---------------
Data References
---------------
.. bibliography:: data_refs.bib
   :all:

-------------------
Software References
-------------------
.. bibliography:: software_refs.bib
   :all:
