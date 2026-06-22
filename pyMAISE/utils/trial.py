import numpy as np
from sklearn.utils.multiclass import type_of_target


def determine_class_from_probabilities(y_pred, y):
    """
    Convert model probability outputs to class predictions.

    Parameters
    ----------
    y_pred: numpy.ndarray
        Probability output from the model.
    y: numpy.ndarray
        Ground-truth labels used to determine the target type.

    Returns
    -------
    y_pred: numpy.ndarray
        Predicted class labels.
    """
    if type_of_target(y) == "binary" or type_of_target(y) == "multiclass":
        # Round to nearest number
        return np.round(y_pred)

    elif type_of_target(y) == "multilabel-indicator":
        assert np.all((y_pred <= 1) & (y_pred >= 0))
        # Assert 0 or 1 for one hot encoding
        return np.where(y_pred == y_pred.max(axis=-1, keepdims=True), 1, 0)
