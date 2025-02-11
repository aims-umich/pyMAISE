==========
User Guide
==========

ML generation in pyMAISE is done in five steps:

1. :ref:`pyMAISE initialization <pymaise_init>`,
2. :ref:`preprocessing <preprocessing>`,
3. :ref:`model initialization <model_init>`,
4. :ref:`hyperparameter tuning <hyperparameter_tun>`,
5. :ref:`postprocessing <postprocessing>`.

Each step and their respective classes are discussed below. While this guide includes the basic functionality of pyMAISE, we recommend you refer to the :doc:`pymaise_api` documentation for specifics. Additionally, for an introductory tutorial, follow the :doc:`benchmarks/mit_reactor` benchmark.

.. _pymaise_init:

----------------------
pyMAISE Initialization
----------------------

To access pyMAISE Python, we must import the library. For functions from the :mod:`pyMAISE.datasets` or :mod:`pyMAISE.preprocessing` modules, we need to import those directly. For example:

.. code-block:: python

   from pyMAISE.datasets import load_fp
   from pyMAISE.preprocessing import correlation_matrix, train_test_split, scale_data
   import pyMAISE as mai

We shorten ``pyMAISE`` to ``mai`` for convenience.

Defining Global Settings
^^^^^^^^^^^^^^^^^^^^^^^^

Every pyMAISE job requires the initialization of some global settings used throughout pyMAISE. These global settings and their defaults include:

- ``problem_type``: the type of problem, either regression or classification, defined using a string or :class:`pyMAISE.ProblemType`,
- ``verbosity = 0``: level of output from pyMAISE,
- ``random_state = None``: the global random seed used for pseudo-random algorithms in ML methods,
- ``num_configs_saved = 5``: the number of hyperparameter configurations to save for each model when tuning,
- ``new_nn_architecture = True``: pyMAISE's neural network tuning architecture was upgraded using `KerasTuner <https://keras.io/keras_tuner/>`_ :cite:`omalley2019kerastuner`, this boolean defines which architecture you use,
- ``cuda_visible_devices = None``: sets the ``CUDA_VISIBLE_DEVICES`` environment variable for `TensorFlow <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_ :cite:`tensorflow2015-whitepaper`,
- ``run_parallel``: run neural network tuning in parallel,
- ``max_models_per_device``: the maximum number of neural networks training on one GPU at once.

To initialize pyMAISE, we define the settings using :meth:`pyMAISE.init`:

.. code-block:: python

   global_settings = mai.init(
      problem_type=mai.ProblemType.REGRESSION,  # Define a regression problem
      random_state=42                           # set random state for pyMAISE
   )

.. _preprocessing:

--------------
Preprocessing
--------------

The :mod:`pyMAISE.preprocessing` offers several methods to read, split, scale, and visualize data before tuning.

Loading Data
^^^^^^^^^^^^

pyMAISE offers several data sets for building and testing ML models. Each of these data sets includes benchmarks in Jupyter Notebooks. These benchmarks have tested classical and neural network models to provide the expected performance for standard ML models. These benchmarks include

- :doc:`benchmarks/mit_reactor`: effect of control blade height on fuel element power,
- :doc:`benchmarks/reactor_physics`: effect of cross section on :math:`k`,
- :doc:`benchmarks/fuel_performance`: effect of fuel parameters on pellet gas production, centerline temperature, surface temperature, and radial displacement,
- :doc:`benchmarks/heat_conduction`: effect of heat conduction parameters on fuel rod centerline temperature,
- :doc:`benchmarks/bwr`: effect of BWR core parameters on :math:`k` and peaking factors,
- :doc:`benchmarks/HTGR_microreactor`: effect of control drum angle on neutron flux,
- :doc:`benchmarks/rod_ejection`: effect of reactor kinetics parameters on max power, burst width, max fuel centerline temperature, and average coolant temperature,
- :doc:`benchmarks/chf`: effect of experimental parameters on critical heat flux (CHF),
- :doc:`benchmarks/anomaly`: effect of high-voltage converter modulator (HVCM) parameters for the Spallation Neutron Source linear accelerators on fault prediction.

Each data set has a load function in the :mod:`pyMAISE.datasets` module. For details, refer to the :doc:`pymaise_api`.

To read your data, you can use the :meth:`pyMAISE.preprocessing.read_csv` function. For input and output data in one file, ``file.csv``:

.. code-block:: python

   from pyMAISE.preprocessing import read_csv
   data, inputs, outputs = read_csv("file.csv", slice(0, x), slice(x, y))

where ``x`` is the end plus one position of the inputs and ``y`` is the end plus one position of the outputs. For data split into two files: ``inputs.csv``, ``outputs.csv``:

.. code-block:: python

   from pyMAISE.preprocessing import read_csv
   data, inputs, outputs = read_csv(["inputs.csv", "outputs.csv"])

Train/Test Splitting Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Using the :meth:`pyMAISE.preprocessing.train_test_split` method, we can split data into training and testing data. For a split of 70% training and 30% testing, we can do

.. code-block:: python

   from pyMAISE.preprocessing import train_test_split
   xtrain, xtest, ytrain, ytest = train_test_split([inputs, outputs], test_size=0.3)

Scaling Data
^^^^^^^^^^^^

Many ML models train best on scaled data. For min-max scaling data, we can use the :meth:`pyMAISE.preprocessing.scale_data` method

.. code-block:: python

   from sklearn.preprocessing import MinMaxScaler
   from pyMAISE.preprocessing import scale_data

   xtrain, xtest, xscaler = scale_data(xtrain, xtest, scaler=MinMaxScaler())
   ytrain, ytest, yscaler = scale_data(ytrain, ytest, scaler=MinMaxScaler())

The ``scaler`` can be anything that has ``fit_transform``, ``transform``, and ``inverse_transform`` methods.

Splitting Time Series Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

For time series data, the :class:`pyMAISE.preprocessing.SplitSequence` class offers to create rolling windows for 2D and 3D time series data. For more information, refer to the :doc:`pymaise_api`.

One-Hot Encoding
^^^^^^^^^^^^^^^^

Some models perform better when the classification data is one-hot encoded. For this use :meth:`pyMAISE.preprocessing.one_hot_encode`.

.. caution:: Outputs must be one-hot encoded for neural network models.

Generating a Correlation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To better understand the correlation between the inputs and the outputs we can plot a correlation matrix using :meth:`pyMAISE.preprocessing.correlation_matrix`.

.. _model_init:

--------------------
Model Initialization
--------------------

pyMAISE supports both classical ML methods and sequential neural networks. For a complete list of supported models, neural network layers, and neural network optimizers refer to the :class:`pyMAISE.Tuner`. These models originate from `scikit-learn <https://scikit-learn.org/stable/index.html>`_ and `Keras <https://keras.io>`_. Please refer to the model documentation for each supported model on the specifics of its algorithm. Each of these models is defined by their hyperparameters, which define algorithmic parameters for training. For dictionaries for the model hyperparameters refer to :doc:`models`.

.. note:: If a classical model, neural network layer, or neural network optimizer is not currently supported, submit an issue at the `pyMAISE github repository <https://github.com/aims-umich/pyMAISE>`_ detailing the object you would like implemented.

To initialize :class:`pyMAISE.Tuner`, we define each model using a list of their keys. These keys are given in the :class:`pyMAISE.Tuner` documentation. For classical models, we define the parameters which remain constant throughout tuning. These hyperparameters are given in subdictionaries under each model key. The default configuration is used if a subdictionary is not provided for a defined model. Here is an example of linear, lasso, and random forest:

.. code-block:: python

   model_settings = {
      "models": ["Linear", "Lasso", "RF"],
      "RF": {
         "n_estimators": 200,
      },
   }

This indicates that we change the ``"n_estimators"`` hyperparameter to 200, the rest are initialized as default.

Some models (right now just Guassian Processing), which will only work when the data is standard scaled. If still selected when not standard scaled, an error will be thrown when postprocessing the data.

Also, some models also have the ability to extend to multi-output without having to pass through the multi-output object in the models array. This is done by turning `multi-output = True` in model settings.
Currently this is supported for gradient boosting, adaboost, and stacking. Please note that when this is done, pyMAISE is actually passing the model through a ``MultiOutput`` object. The same quirks about ``MultiOutput`` Hyperparameter Tuning (as discussed in the next section) still apply.

For neural networks, we define both the hyperparameters that remain constant during tuning and those that change. The hyperparameters that change are set using :class:`pyMAISE.Int`, :class:`pyMAISE.Float`, :class:`pyMAISE.Choice`, :class:`pyMAISE.Boolean`, and :class:`pyMAISE.Fixed`. These hyperparameters are set within the ``"structural_params"``, ``"optimizer"``, ``"compile_params"``, and ``"fitting_params"`` keys within the models subdictionary. For each neural network layer, we can also define the ``"sublayer"``, ``"wrapper"``, and ``"num_layers"`` hyperparameters. For example, here is a dense feedforward neural network:

.. code-block:: python

   model_settings = {
       "models": ["FNN"],
       "FNN": {
           "structural_params": {
               "Dense_hidden": {
                   "num_layers": mai.Int(min_value=0, max_value=3),
                   "units": mai.Int(min_value=25, max_value=400),
                   "activation": "relu",
                   "kernel_initializer": "normal",
                   "sublayer": mai.Choice(["Dropout_hidden", "None"]),
                   "Dropout_hidden": {
                       "rate": mai.Float(min_value=0.4, max_value=0.6),
                   },
               },
               "Dense_output": {
                   "units": ytrain.shape[-1],
                   "activation": "linear",
                   "kernel_initializer": "normal",
               },
           },
           "optimizer": "Adam",
           "Adam": {
               "learning_rate": mai.Float(min_value=1e-5, max_value=0.001),
           },
           "compile_params": {
               "loss": "mean_absolute_error",
               "metrics": ["mean_absolute_error"],
           },
           "fitting_params": {
               "batch_size": mai.Choice([8, 16, 32]),
               "epochs": 50,
               "validation_split": 0.15,
           },
       },
   }

.. caution:: The layers within ``"structural_params"`` must be named differently with their keyword present. For example, ``"Dense_input"``, ``"Dense_hidden"``, ``"Dense_output"``. Here ``"Dense"`` is the keyword pyMAISE needs.

With this dictionary of models and parameters, we initialize the :class:`pyMAISE.Tuner`:

.. code-block:: python

   tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)

.. _hyperparameter_tun:

----------------------
Hyperparameter Tuning
----------------------

With all the models of interest initialized in the :class:`pyMAISE.Tuner`, we can begin hyperparameter tuning. pyMAISE supports three types of search methods for classical models (grid, random, and Bayesian search) and four types for neural networks (grid, random, Bayesian, and hyperband search). For the classical model methods we define the search space using the array, distribution or `skopt.space.space <https://scikit-optimize.github.io/stable/modules/classes.html#module-skopt.space.space>`_ for each hyperparameter we want to tune. For neural networks, we do not need to redefine the search space. For specifics on the methods and their arguments, refer to the :doc:`pymaise_api`.

All methods include a ``cv`` argument, which defines the cross-validation used during tuning. If an integer is given, then the data set is either split with `sklearn.model_selection.KFold <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold>`_ or `sklearn.model_selection.StratifiedKFold <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_ depending on the data set's target type. We can also pass any cross-validation callable that includes a ``split`` method.

Grid Search with Classical Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Grid search evaluates all possible combinations of a given parameter space. To define the parameter search space for classical models we define a dictionary of Numpy arrays or lists for each parameter of interest. For the classical models defined in the above section, we can define

.. code-block:: python

   grid_search_spaces = {
      "lasso": {"alpha": np.linspace(0.0001, 5, 20)},
      "rforest": {
          "max_features": [None, "sqrt", "log2", 2, 4, 6],
      },
   }

This dictionary is then passed to the grid search tuning function:

.. code-block:: python

   grid_search_configs = tuner.grid_search(
      param_spaces=grid_search_spaces,
   )

Which will run the grid search. Notice that a ``Linear`` search space was not defined; in this case, the model's parameters are returned for postprocessing, and no tuning takes place.

With the new pyMAISE version, we have added stacking and multi-output, which both add a layer of complexity to getting to model parameters.
When using stacking models, hyperparameters are accessed through the format ``{name}__{property}``, where {name} is the identifier for each base estimator in the stack, and {property} is the specific hyperparameter. For example, if you want to set the alpha parameter for a elastic net estimator in a stacking model, you would use ``EN__alpha``.
For multi-output tasks with a single model, you can access properties directly using the format ``estimator__{property}``, as there is only one main estimator for all outputs. This allows for a simplified tuning structure in multi-output models, where specifying parameters does not require naming individual base estimators.
Please note that if multi-output is turned on for models when setting model parameters, the same wrapping is applied and you will need to access model parameters through ``estimator__{property}``.

**Example**

If tuning a stacking model, parameters might be set as:

.. code-block:: python

   stacking_search_spaces = {
      "Lasso__alpha": np.linspace(0.0001, 5, 20),
      "RF__max_depth": [5, 10, 15],
   }

For a multi-output estimator (this example has a wrapped elsatic net estimator):

.. code-block:: python

   multi_output_search_space = {
      "estimator__max_depth": np.linspace(0.0001, 5, 20),
   }

As previously stated, stacking has the option to extend to multi-output without explicitly making a multi-output model. This will require double nesting to get to model parameters as ``estimator__{name}__{property}``. This is so we can access the Stacking object (and it's models) underneath the multi-output wrapper.

Random Search with Classical Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random search evaluates the hyperparameter configurations sampled from distributions. These distributions can be a list or a callable with an ``rvs`` method. In the pyMAISE Jupyter Notebooks, we use the distributions from `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. For example, for linear, lasso, and random forest we can do

.. code-block:: python

   from scipy.stats import uniform

   random_search_spaces = {
      "lasso": {
          # Uniform distribution for alpha between 0.0001 - 0.01
          "alpha": scipy.stats.uniform(loc=0.0001, scale=0.0099),
      },
      "rforest": {
          "max_features": [None, "sqrt", "log2", 2, 4, 6],
      },
   }

We can then define the models, number of iterations, cross-validation, and other parameters in :meth:`pyMAISE.Tuner.random_search`:

.. code-block:: python

   random_search_configs = tuner.random_search(
      param_spaces=random_search_spaces,
      n_iter=200,
      cv=5,
   )

Bayesian Search with Classical Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayesian search uses results from prior hyperparameter configurations to inform the next iteration of hyperparameters. This attempts to converge on the optimal hyperparameter configuration using a Gaussian process surrogate function to predict the next parameter configuration. For :meth:`pyMAISE.Tuner.bayesian_search` we define the search space using `skopt.space.space <https://scikit-optimize.github.io/stable/modules/classes.html#module-skopt.space.space>`_ parameters. For linear, lasso, and random forest we can do

.. code-block:: python

   from skopt.space.space import Integer, Real

   bayesian_search_spaces = {
      "lasso": {
          "alpha": Real(0.0001, 0.01),
      },
      "rforest": {
          "max_features": Integer(1, 10),
      },
   }

We can then pass this to :meth:`pyMAISE.Tuner.bayesian_search`:

.. code-block:: python

   bayesian_search_configs = tuner.bayesian_search(
      param_spaces=bayesian_search_spaces,
      n_iter=50,
   )

where we pass the parameter spaces, the number of iterations, and other parameters. Bayesian search will then sample between the limits defined in ``bayesian_search_spaces``.

Convergence Plots
^^^^^^^^^^^^^^^^^

For each of the search methods you can plot a convergence plot using the :meth:`pyMAISE.Tuner.convergence_plot` function; however, this is more appealing for Bayesian search as it shows how the kernel converges to the optimal hyperparameter configuration with each step. To plot a specific model such as a feedforward neural network named ``"FNN"`` run

.. code-block:: python

   tuner.convergence_plot(model_types="FNN")

.. _postprocessing:

--------------
Postprocessing
--------------

With our top :attr:`pyMAISE.Settings.num_configs_saved` models we can pass these to the ``PostProcessor`` class for model comparison and testing. To do so we provide the scaled data, configuration(s), and the yscaler:

.. code-block:: python

   postprocessor = mai.PostProcessor(
      data=(xtrain, xtest, ytrain, ytest),
      models_list=[random_search_configs, bayesian_search_configs],
      yscaler=yscaler
   )

Additionally, we can pass a dictionary similar to ``model_settings`` of updated model settings to the ``new_model_settings`` parameter such as an increase in epochs for the final neural network models. With our :class:`pyMAISE.PostProcessor` initialized we can begin evaluating our models.
Please note that if Gaussian Processing was selected and standard scaling was not applied, an error will be thrown in this step.

Performance Metrics
^^^^^^^^^^^^^^^^^^^

The :meth:`pyMAISE.PostProcessor.metrics` function evaluates performance metrics for the training and testing predictions of each model. :meth:`pyMAISE.PostProcessor.metrics` by default evaluates

- r-squared: :math:`\text{R}^2 = 1 - \frac{\sum_{i = 1}^{n}(y_i - \hat{y_i})^2}{\sum_{i = 1}^{n}(y_i - \bar{y}_i)^2}`,
- mean absolute error: :math:`\text{MAE} = \frac{1}{n}\sum_{i = 1}^{n}|y_i - \hat{y}_i|`,
- mean absolute percentage error :math:`\text{MAPE} = \frac{100}{n}\sum_{i = 1}^{n}\frac{|y_i - \hat{y}_i|}{\text{max}(\epsilon, |y_i|)}`,
- root mean squared error: :math:`\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i = 1}^n(y_i - \hat{y}_i)^2}`,
- root mean square percentage error: :math:`\text{RMSPE} = 100\sqrt{\frac{1}{n}\sum_{i = 1}^n\Big(\frac{y_i - \hat{y}_i}{\text{max}(\epsilon, |y_i|)}\Big)^2}`

for regression problems where :math:`y` is the actual outcome, :math:`\hat{y}` is the model predicted outcome, :math:`\bar{y}` is the average outcome, :math:`\epsilon` is an arbitrarily small positive number to avoid undefined values, and :math:`n` is the number of observations. For classification problems the defaults are

- accuracy: :math:`\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}`,
- recall: :math:`\text{Recall} = \frac{\text{True positives}}{\text{True positives} + \text{False negatives}}`,
- precision: :math:`\text{Precision} = \frac{\text{True positives}}{\text{True positives} + \text{False positives}}`,
- F1: :math:`\text{F1} = 2\frac{\text{Precision}\times\text{Recall}}{\text{Precision} + \text{Recall}}`,

Additionally, we can supply our own metrics to the ``metrics`` as callables. We can choose how the DataFrame is sorted, whether the features are averaged or only the metrics for one feature are computed, and which models to show. With this information, we can compare the performance of each of our models on our data set.

Performance Visualized
^^^^^^^^^^^^^^^^^^^^^^

To visualize the performance of each of these models we can use :meth:`pyMAISE.PostProcessor.diagonal_validation_plot`, :meth:`pyMAISE.PostProcessor.validation_plot`, and :meth:`pyMAISE.PostProcessor.nn_learning_plot`. The first two methods provide a comparison of the predicted outcomes versus the actual and :meth:`pyMAISE.PostProcessor.nn_learning_plot` provides a neural network learning curve for comparing training and validation performance. You can plot neural network structures with :meth:`pyMAISE.PostProcessor.nn_network_plot`.

For classification problems we can create a confusion matrix using :meth:`pyMAISE.PostProcessor.confusion_matrix`.

Other Postprocessing Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, the :class:`pyMAISE.PostProcessor` is equipped with several additional methods for analysis. These include

- :meth:`pyMAISE.PostProcessor.get_params`: get the parameter configurations from a specific model,
- :meth:`pyMAISE.PostProcessor.get_model`: get the model wrapper,
- :meth:`pyMAISE.PostProcessor.get_predictions`: get the training and testing predictions from a specific model,
- :meth:`pyMAISE.PostProcessor.print_model`: print a models tuned hyperparameters,
- :meth:`pyMAISE.PostProcessor.save_models`: save the best performing models.

---------------
pyMAISE Testing
---------------

pyMAISE includes a regression and unit testing suite that is run with each push to the repository. These tests ensure the functionality and validity of the results generated with pyMAISE. Regression tests include performance metric checks based on benchmarked scripts. These scripts are provided in the ``scripts`` directory within pyMAISE.
