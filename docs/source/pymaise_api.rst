pyMAISE API Reference
=====================

pyMAISE offers ML generation and evaluation using five processes:

1. :ref:`settings <settings_api>`,
2. :ref:`data sets <datasets_api>`,
3. :ref:`preprocessing <preprocessing_api>`,
4. :ref:`hyperparameter tuning <tuning_api>`,
5. :ref:`postprocessing <postprocessing_api>`.

These processes allow you to define models of interest,
tune them to your or one of the data sets within pMAISE, and assess
their performance. Use this page for further information
on a specific function or class. Feel free to click on the functions or
classes to see their docstrings outlining arguments, outputs, and use.

.. _settings_api:

Settings
--------

Every pyMAISE script begins with defining some global
settings used throughout the package. This is done through the
:meth:`pyMAISE.init` function, where you can define the :class:`pyMAISE.ProblemType`,
level of output from pyMAISE, the number of hyperparameter configurations saved
for each model, and more.

.. rubric:: Functions

.. autosummary::
   :toctree: stubs
   :nosignatures:

   pyMAISE.init

.. rubric:: Classes

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: class.rst

   pyMAISE.ProblemType

.. _datasets_api:

Data Sets
---------

pyMAISE includes several benchmark data sets used in the
:ref:`benchmark notebooks <examples>`, which also serve as example notebooks when
using pyMAISE. These data sets derive from several nuclear engineering
applications and originate from literature. For information on the data sets past
what is given here and in the function's documentation, refer to the :ref:`data \
references <data_refs>`.

Each of these load functions exits under the :mod:`pyMAISE.datasets` module.
To import the MIT reactor data, for example, we can do the following:

.. code-block:: python

   from pyMAISE.datasets import load_MITR

From pyMAISE, we cannot directly access the data set load functions.

.. rubric:: Functions

.. autosummary::
   :toctree: stubs
   :nosignatures:

   pyMAISE.datasets.load_MITR
   pyMAISE.datasets.load_xs
   pyMAISE.datasets.load_fp
   pyMAISE.datasets.load_heat
   pyMAISE.datasets.load_BWR
   pyMAISE.datasets.load_HTGR
   pyMAISE.datasets.load_rea
   pyMAISE.datasets.load_chf
   pyMAISE.datasets.load_loca

.. _preprocessing_api:

Preprocessing
-------------

Using one of the provided data sets or your own, you can utilize the preprocessing
module to read data from a CSV file, split it into training and testing data,
and scale it. This module also offers methods specific to
classification problems, such as one hot encoding. You can use
:class:`pyMAISE.preprocessing.SplitSequence` to create rolling windows of
your time series data and create a correlation matrix.

Like the data sets, the :mod:`pyMAISE.preprocessing` module functions cannot
be accessed from pyMAISE. So we import functions using:

.. code-block:: python

   from pyMAISE.preprocessing import train_test_split, scale_data


.. warning::
   For multiclass classification problems, the output must be one hot-encoded
   for neural network models.

.. rubric:: Functions

.. autosummary::
   :toctree: stubs
   :nosignatures:

   pyMAISE.preprocessing.read_csv
   pyMAISE.preprocessing.train_test_split
   pyMAISE.preprocessing.scale_data
   pyMAISE.preprocessing.one_hot_encode
   pyMAISE.preprocessing.correlation_matrix

.. rubric:: Classes

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: class.rst

   pyMAISE.preprocessing.SplitSequence

.. _tuning_api:

Tuning
------

The :class:`pyMAISE.Tuner` allows you to specify models for hyperparameter
tuning and the tuning method you'd like to use. Additionally, the class
offers the :meth:`pyMAISE.Tuner.convergence_plot` method for plotting
the tuning methods results.

.. rubric:: Classes

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: class.rst

   pyMAISE.Tuner

.. _hyperparameters_api:

Hyperparameters
^^^^^^^^^^^^^^^

When initializing a neural network model, you can use these classes so
pyMAISE knows which hyperparameters you would like to tune.

.. rubric:: Classes

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: class.rst

   pyMAISE.Int
   pyMAISE.Float
   pyMAISE.Choice
   pyMAISE.Boolean
   pyMAISE.Fixed

.. _postprocessing_api:

Postprocessing
---------------

Following the tuning of the specified models. You can use the
:class:`pyMAISE.PostProcessor` to access the performance of your models.
This offers the ability to fit the models with different hyperparameters such
as more epochs and access their performance metrics on both training and
testing data. There are additional getters and visualization tools for
in-depth evaluation.

.. rubric:: Classes

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: class.rst

   pyMAISE.PostProcessor
