import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split as sklearn_train_test_split

import pyMAISE.settings as settings
from pyMAISE.utils import _try_clear


def read_csv(
    path,
    input_slice=None,
    output_slice=None,
    header="infer",
):
    """
    Read comma-separated values (csv) file into ``xarray.DataArray``. This is used for
    an entire data set and/or its inputs and outputs.

    Parameters
    ----------
    path : str or list of str
        The path to the single input file containing both inputs and outputs
        or a list of two paths, the first to the inputs and the second to
        the outputs.
    input_slice : slice or None, default=None
        Index slice of inputs from a data set containing both inputs and outputs
        for a single string ``path``.
    output_slice : slice or None, default=None
        Index slice of outputs from a data set containing both inputs and outputs
        for a single string ``path``.
    header : int, 'infer', or None, default=`infer`
        Row number(s) containing column labels and marking the start of the data.
        Used in
        `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/\
        api/pandas.read_csv.html>`_
        :cite:`mckinney2010data`.

    Returns
    -------
    data : xarray.DataArray
        The input and output data given by the files.
    inputs : xarray.DataArray
        The input data produced from either slicing ``data`` with ``input_slice``
        or from reading the first file path given in ``path``. If a single path
        is given and ``input_slice`` and ``output_slice`` is none then only
        ``data`` is returned.
    outputs : xarray.DataArray
        The output data is produced in the same fashion as ``inputs`` using
        ``output_slice``. If a list of file paths is given for ``data`` then this
        corresponds to the second.
    """
    print("Reading data from", path)

    if isinstance(path, str):
        # Read in data from path
        data = (
            pd.read_csv(path, header=header)
            .to_xarray()
            .to_array()
            .transpose(..., "variable")
        )

        if input_slice is None and output_slice is None:
            _try_clear()
            return data
        else:
            assert input_slice is not None and output_slice is not None
            # Separate data based on slices
            inputs = data.isel(variable=input_slice)
            outputs = data.isel(variable=output_slice)

            data = xr.concat([inputs, outputs], dim=data.dims[-1])

            _try_clear()
            return data, inputs, outputs

    elif isinstance(path, list):
        # Make sure only two paths are given
        assert len(path) == 2

        # Assuming the first path is the input and second is
        # output data
        inputs = (
            pd.read_csv(path[0], header=header)
            .to_xarray()
            .to_array()
            .transpose(..., "variable")
        )
        outputs = (
            pd.read_csv(path[1], header=header)
            .to_xarray()
            .to_array()
            .transpose(..., "variable")
        )

        # Join input and output
        data = inputs.combine_first(outputs)

        _try_clear()
        return data, inputs, outputs

    else:
        raise TypeError(f"path = {path} is neither a string or list of strings")


def one_hot_encode(data, **kwargs):
    """
    One hot encode multiclass classification data. This is required for
    training neural network models. This function utilizes
    `pandas.get_dummies <https://pandas.pydata.org/pandas-docs/stable/reference/\
    api/pandas.get_dummies.html>`_ :cite:`mckinney2010data`.

    Parameters
    ----------
    data: xarray.DataArray
        All data to be one hot encoded.
    dtype: dtype, default=float
        The data type.

    Returns
    -------
    dummified_data: xarray.DataArray
        One hot encoded data.
    """
    assert isinstance(data, xr.DataArray)
    kwargs["dtype"] = kwargs.get("dtype", float)

    # Convert xarray.DataArray to pandas.DataFrame
    df = data.to_dataset(dim=data.dims[-1]).to_dataframe()

    # One hot encoding
    df = pd.get_dummies(df, **kwargs)

    # Return one hot encoded xarray.DataArray
    return (
        df.to_xarray()
        .to_array()
        .rename({"variable": data.dims[-1]})
        .transpose(data.dims[0], ..., data.dims[-1])
    )


class SplitSequence:
    """
    Split sequence function for rolling windows of time series data. Using a rolling
    windows, 2D-time series data of dimensions (time steps, features) is split according
    to the features defined in ``sequence_inputs``, ``sequence_outputs``, and the
    windows width and positional information. This results in a 3D input data set and a
    2D or 3D output data set.

    If the data set is 3D, then rolling windows are applied to the sequences specified
    in ``sequence_inputs`` and ``sequence_outputs`` resulting in a 4D array. The
    features and windows (the last two dimensions) are combined to create a 3D data set.
    Features without rolling windows are specified in ``feature_inputs`` and
    ``feature_outputs`` and are concatenated to get 3D input and output
    ``xarray.DataArray`` objects.

    Parameters
    ----------
    input_steps: int
        The window size or number of time steps for each input sample.
    output_steps: int
        The window size or number of time steps for each output sample.
    output_position: int
        The position to start the output window relative to the position of the final
        time step in the input window. If the last time step in the input window is
        at index five and ``output_position=1``, then the output window begins at index
        six.
    sequence_inputs: None or array/list of int or str
        Corresponds to the features (last dimension of ``data``) taken
        for inputs. If ``None`` then the entire data set is used for inputs.
    sequence_outputs: None or array/list of int or str
        Corresponds to the labels (last dimension of ``data``) that are taken for
        outputs. If ``None`` then the entire data set is used for outputs.
    const_inputs: None or array/list of int or str
        The features concatenated to the input windows that are not used in
        rolling windows. This is only used when ``data`` is 3D.
    const_outputs: None or array/list of int or str
        The labels concatenated to the input windows that are not used in
        rolling windows. This is only used when ``data`` is 3D.

    Examples
    --------

    Using the 2D LOCA data set, we demonstrate rolling windows on the
    perturbed data.

    >>> from pyMAISE import datasets, preprocessing
    >>> _, perturbed = datasets.load_loca(stack_series=True)
    >>> perturbed.shape
    (1600000, 44)
    >>> sequence_outputs = [
            "Pellet Cladding Temperature",
            "Core Pressure",
            "Water Level",
            "Break Flow Rate"
        ]
    >>> split_sequences = preprocessing.SplitSequence(
            4,
            1,
            1,
            sequence_outputs=sequence_outputs
        )
    >>> perturbed_input, perturbed_output = split_sequences.split(perturbed)
    >>> perturbed_input.shape
    (1599996, 4, 44)
    >>> perturbed_output.shape
    (1599996, 4)

    Alternatively, we can use the 3D perturbed LOCA data, specify the four
    sequential features as inputs and outputs and then add the time-independent
    features.

    >>> from pyMAISE import datasets, preprocessing
    >>> _, perturbed = datasets.load_loca(stack_series=False)
    >>> perturbed.shape
    (4000, 400, 44)
    >>> split_sequences = preprocessing.SplitSequence(
            input_steps=4,
            output_steps=1,
            output_position=1,
            sequence_inputs=range(-4, 0),
            sequence_outputs=range(-4, 0),
            const_inputs=range(40),
        )
    >>> perturbed_input, perturbed_output = split_sequences.split(perturbed)
    >>> perturbed_input.shape
    (4000, 396, 56)
    >>> perturbed_output.shape
    (4000, 396, 4)
    """

    def __init__(self, input_steps, output_steps, output_position, **kwargs):
        self._input_steps = input_steps
        self._output_steps = output_steps
        self._output_position = output_position

        # Defaults
        self._sequence_inputs = kwargs.get("sequence_inputs", None)
        self._sequence_outputs = kwargs.get("sequence_outputs", None)
        self._const_inputs = kwargs.get("const_inputs", None)
        self._const_outputs = kwargs.get("const_outputs", None)

    def split(self, data):
        """
        Run rolling windows.

        Parameters
        ----------
        data: xarray.DataArray
            A data set that includes both input and output sequence data.
            This data can be either 2 or 3-dimensional.

        Returns
        -------
        split_input : xarray.DataArray
            The 3D data set of input data with dimensions (samples, time steps,
            features).
        split_output : xarray.DataArray
            The 3D or 2D data set of output data with either dimensions
            (samples, time steps, labels) or (samples, labels). If ``output_steps=1``
            then the time steps dimension is removed.
        """
        # Assert our data is either 2D or 3D
        assert isinstance(data, xr.DataArray)
        assert len(data.shape) > 1 and len(data.shape) < 4

        # Function for checking index type, turns lists of str to array of ints
        def type_based_index(index_list):
            if index_list is None:
                return data
            if isinstance(index_list[0], str):
                return data.sel(**{data.dims[-1]: index_list})
            else:
                return data.isel(**{data.dims[-1]: index_list})

        inputs = type_based_index(self._sequence_inputs)
        outputs = type_based_index(self._sequence_outputs)

        # Get feature names
        input_feature_strs = inputs.coords[inputs.dims[-1]].values
        output_feature_strs = outputs.coords[outputs.dims[-1]].values

        # Check the number of dimensions in the given data set
        # If 2D then assume (timesteps, features)
        # If 3D then assume (samples, timesteps, features)
        temporal_index = 0
        if len(data.shape) == 3:
            temporal_index = 1

        # Trim input array based on output window and position and create
        # rolling window
        x = np.lib.stride_tricks.sliding_window_view(
            np.take(
                inputs.values,
                np.arange(
                    inputs.shape[temporal_index]
                    - self._output_position
                    - self._output_steps
                    + 1
                ),
                axis=temporal_index,
            ),
            self._input_steps,
            axis=temporal_index,
        )

        # Trim output array based on input window and output position
        # and create rolling window
        y = np.lib.stride_tricks.sliding_window_view(
            np.take(
                outputs.values,
                np.arange(
                    self._input_steps + self._output_position - 1,
                    outputs.shape[temporal_index],
                ),
                axis=temporal_index,
            ),
            self._output_steps,
            axis=temporal_index,
        )

        # Swap window and feature axes to make feature last dimension
        x = np.swapaxes(x, -2, -1)
        y = np.swapaxes(y, -2, -1)

        # If our windowed arrays are 4D then compress windows and features
        if temporal_index == 1:
            # Expand names
            windowed_features = []
            windowed_labels = []
            for name in input_feature_strs:
                windowed_features = windowed_features + [
                    f"{name}_{timestep}" for timestep in range(x.shape[2])
                ]
            for name in output_feature_strs:
                windowed_labels = windowed_labels + [
                    f"{name}_{timestep}" for timestep in range(y.shape[2])
                ]
            input_feature_strs = windowed_features
            output_feature_strs = windowed_labels

            # Stack time steps into features dimension
            x = x.reshape((x.shape[0], x.shape[1], -1))
            y = y.reshape((y.shape[0], y.shape[1], -1))

            # Add features that were not used in windows
            if self._const_inputs is not None:
                const_inputs = type_based_index(self._const_inputs)
                input_feature_strs = (
                    list(const_inputs.coords[const_inputs.dims[-1]].values)
                    + input_feature_strs
                )
                x = np.concatenate(
                    (
                        np.take(
                            const_inputs.values,
                            np.arange(x.shape[temporal_index]),
                            axis=temporal_index,
                        ),
                        x,
                    ),
                    axis=2,
                )
            if self._const_outputs is not None:
                const_outputs = type_based_index(self._const_outputs)
                output_feature_strs = (
                    list(const_outputs.coords[const_outputs.dims[-1]].values)
                    + output_feature_strs
                )
                y = np.concatenate(
                    (
                        np.take(
                            const_outputs.values,
                            np.arange(x.shape[temporal_index]),
                            axis=temporal_index,
                        ),
                        y,
                    ),
                    axis=2,
                )

        # Create input xarray.DataArray
        split_inputs = xr.DataArray(
            x,
            coords={
                "samples": np.arange(x.shape[0]),
                inputs.dims[temporal_index]: np.arange(x.shape[1]),
                inputs.dims[-1]: input_feature_strs,
            },
        )

        # Create output xarray.DataArray, if we have only one output_step
        # then we ommit the time dimension
        split_outputs = (
            xr.DataArray(
                y.reshape(-1, y.shape[-1]),
                coords={
                    "samples": np.arange(y.shape[0]),
                    outputs.dims[-1]: output_feature_strs,
                },
            )
            if y.shape[1] == 1
            else xr.DataArray(
                y,
                coords={
                    "samples": np.arange(y.shape[0]),
                    inputs.dims[temporal_index]: np.arange(y.shape[1]),
                    outputs.dims[-1]: output_feature_strs,
                },
            )
        )

        return split_inputs, split_outputs


def train_test_split(
    data,
    test_size=0.3,
):
    """
    Split data into training and testing data sets.

    Parameters
    ----------
    data: xarray.DataArray or list of 2 xarray.DataArray[s]
        Data to be split. Assumes the first dimension is the sample dimension.
    test_size: float between 0 and 1
        Percentage of the data used for testing.

    Returns
    -------
    split_data: tuple of xarray.DataArray
        If an ``xarray.DataArray`` was given for ``data`` then split_data consists of
        ``(data_train, data_test)``. If a list of 2 ``xarray.DataArray`` objects,
        ``[x, y]``, then split_data consists of ``(x_train, x_test, y_train, y_test)``.
    """
    if isinstance(data, xr.DataArray):
        data = [data]
    if not all(
        (isinstance(x, xr.DataArray) and x.shape[0] == data[0].shape[0] for x in data)
    ):
        raise RuntimeError(
            "data must either be an xarray.DataArray or a list of "
            + "xarray.DataArray with the same number of samples (dimension 0)"
        )

    samples_idx = np.arange(data[0].shape[0])
    train_idx, test_idx = sklearn_train_test_split(
        samples_idx,
        test_size=test_size,
        random_state=settings.values.random_state,
    )

    # Split into train and test using indices
    split_data = []
    for array in data:
        split_data.append(array[train_idx,])
        split_data.append(array[test_idx,])

    return tuple(split_data)


def scale_data(train_data, test_data, scaler):
    """
    Scale training and testing data using the scaler provided. This method returns the
    fit scalar which can be used to scale any additional data.

    Example
    -------

    Given the following 2D ``xarray.DataArray``s of shape (samples,
    features/labels), ``xtrain`` and ``xtest``, we can apply the
    ``pyMAISE.preprocessing.scale_data()`` method,

    .. code:: python

       from pyMAISE.preprocessing import scale_data
       from sklearn.preprocessing import MinMaxScaler

       xtrain, xtest, xscaler = scale_data(xtrain, xtest, MinMaxScaler)

    We can then scale an additional dataset, ``xvalid``, that matches the
    format of ``xtrain`` and ``xtest`` by running

    .. code:: python

       xvalid.values = xscaler.transform(xvalid.values)


    Parameters
    ----------
    train_data: xarray.DataArray
        Training data.
    test_data: xarray.DataArray
        Testing data.
    scaler: callable
        An object with ``fit_transform`` and ``transform`` methods such as
        `min-max scaler from sklearn <https://scikit-learn.org/stable/\
        modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_
        :cite:`scikit-learn`.

    Returns
    -------
    train_data: xarray.DataArray
        Scaled training data.
    test_data: xarray.DataArray
        Scaled testing data based on ``scaler`` fit on ``train_data``.
    scaler: callable
        The scaler given, fit and used to scale the given data.
    """
    # Check that our train_data and test_data are xarray.DataArray[s]
    if not (
        isinstance(train_data, xr.DataArray) and isinstance(test_data, xr.DataArray)
    ):
        raise TypeError("train_data and test_data must be type xarray.DataArray")

    # Check the dimensions of train_data and test_data for all but samples dimension
    if not (train_data.shape[1:] == test_data.shape[1:]):
        raise RuntimeError(
            "train_data and test_data must have the same number of\n"
            + "dimensions and the same size in all but in dimension 0"
        )

    # Transform data
    train_data.values = scaler.fit_transform(
        train_data.values.reshape(-1, train_data.shape[-1])
    ).reshape(train_data.shape)
    test_data.values = scaler.transform(
        test_data.values.reshape(-1, test_data.shape[-1])
    ).reshape(test_data.shape)

    # Return data
    return train_data, test_data, scaler


def correlation_matrix(
    data,
    method="pearson",
    min_periods=1,
    fig=None,
    ax=None,
    colorbar=True,
    annotations=False,
):
    """
    Create a correlation matrix for a data set. This function uses
    `pandas.DataFrame.corr <https://pandas.pydata.org/pandas-docs/stable/\
    reference/api/pandas.DataFrame.corr.html>`_ :cite:`mckinney2010data`,
    for ``method`` and ``min_periods`` please refer to their documentation.

    Parameters
    ----------
    data: xarray.DataArray
        Raw data.
    fig: matplotlib.figure or None, default=None
        Figure object. If ``None`` then one is created.
    ax: matplotlib.pyplot.axis or None, default=None
        Axis object. If ``None`` then one is created.
    colorbar: bool, default=True
        Add colorbar to plot.
    annotations: bool, default=False
        Add annotations to elements.

    Returns
    -------
    fig: matplotlib.figure
        Created or provided figure.
    ax: matplotlib.pyplot.axis
        Created or provided axis.
    """
    # Create correlation matrix DataFrame
    corr = data.to_dataset(dim=data.dims[-1]).to_dataframe().corr(method, min_periods)

    # Get figure and axis object if not provided
    if ax is None and fig is None:
        fig, ax = plt.subplots()

    # Create heatmap
    im = ax.imshow(corr)

    # Set x and y ticks to column headers
    ax.set_xticks(np.arange(corr.shape[0]), labels=corr.columns, rotation=65)
    ax.set_yticks(np.arange(corr.shape[1]), labels=corr.columns, rotation=0)

    # Add colorbar if colorbar == True
    if colorbar:
        fig.colorbar(im)

    # Add annotations of each value in square if annotations == True
    if annotations:
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(
                    j,
                    i,
                    round(corr.to_numpy()[i, j], 2),
                    ha="center",
                    va="center",
                    color="black",
                )

    return (fig, ax)
