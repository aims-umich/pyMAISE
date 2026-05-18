import numpy as np


def validate_trial_results(results, objective, func_name):
    """Validate that a trial result is a numeric value or a dict containing the objective key."""
    if isinstance(results, list):
        for elem in results:
            validate_trial_results(elem, objective, func_name)
        return

    # Single numeric value
    if isinstance(results, (int, float, np.floating)):
        return

    if results is None:
        raise TypeError(
            f"The return value of {func_name} is None. "
            "Did you forget to return the metrics?"
        )

    if isinstance(results, dict):
        if objective not in results:
            raise ValueError(
                f"Expected the returned dictionary from {func_name} to have "
                f"the specified objective, {objective}, as one of the keys. "
                f"Received: {results}."
            )
        return

    raise TypeError(
        f"Expected the return value of {func_name} to be "
        "one of float, dict, or a list of one of these types. "
        f"Received return value: {results} of type {type(results)}."
    )
