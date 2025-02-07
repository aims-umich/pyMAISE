##############
 Installation
##############

.. _prerequisites:

***************
 Prerequisites
***************

pyMAISE currently supports Python 3.9, 3.10, and 3.11. Below are the required
and optional dependencies for pyMAISE. There is no need to install these yourself as
pyMAISE will also install all needed dependencies.

.. admonition:: Required
   :class: error

   -  `NumPy <https://numpy.org/>`_
   -  `pandas <https://pandas.pydata.org/>`_
   -  `Xarray <https://docs.xarray.dev/en/stable/index.html>`_
   -  `scikit-learn <https://scikit-learn.org/stable/index.html>`_
   -  `scikit-optimize <https://scikit-optimize.github.io/stable/>`_
   -  `Keras <https://keras.io>`_
   -  `KerasTuner <https://keras.io/keras_tuner/>`_
   -  `TensorFlow <https://tensorflow.org>`_
   -  `SciKeras <https://adriangb.com/scikeras/stable/>`_
   -  `Matplotlib <https://matplotlib.org/stable/>`_
   -  `tqdm <https://tqdm.github.io/>`_
   -  `pydot <https://github.com/pydot/pydot>`_
   -  `graphviz <https://graphviz.org/>`_
   -  `slicer <https://slicer.readthedocs.io/>`_
   -  `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_
   -  `numba <https://numba.pydata.org/>`_

.. admonition:: Optional
   :class: note

   -  `Jupyter <https://jupyter.org/>`_
   -  `OpenCV <https://opencv.org/>`_
   -  `SciPy <https://scipy.org/>`_

*****
 Pip
*****

Install through Pip by running:

.. code:: sh

   pip install pyMAISE

To install a specific version of pyMAISE, run:

.. code:: sh

   pip install pyMAISE==<version>

Released versions and a discussion of the changes are listed in the
:ref:`versions`. Only stable versions are listed on PyPI. For other
versions or the latest features, install pyMAISE from the source. For
running or creating pyMAISE benchmarks, install the additional dependencies:

.. code:: sh

   pip install "pyMAISE[benchmarks]"

*************
 From Source
*************

For the latest features in development or access to benchmark, install
pyMAISE from source. You can install from source without cloning the environment
by running:

.. code:: sh

   pip install git+https://github.com/myerspat/pyMAISE.git

Alternatively, you can clone the repository using ``git``:

.. code:: sh

   git clone https://github.com/myerspat/pyMAISE.git
   cd pyMAISE/

For a specific version, then checkout the branch:

.. code:: sh

   git checkout <version>

Then install pyMAISE through pip:

.. code:: sh

   pip install .

For benchmarking, install the additional dependencies:

.. code:: sh

   pip install ".[benchmarks]"

For pyMAISE developers, we recommend using the ``-e`` option and installing
the ``dev`` extension:

.. code:: sh

   pip install -e ".[dev]"
