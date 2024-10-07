# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for Tuner class."""


import numpy as np

from keras_tuner import errors


def validate_trial_results(results, objective, func_name):
    if isinstance(results, list):
        for elem in results:
            validate_trial_results(elem, objective, func_name)
        return

    # Single value.
    if isinstance(results, (int, float, np.floating)):
        return

    # None
    if results is None:
        raise errors.FatalTypeError(
            f"The return value of {func_name} is None. "
            "Did you forget to return the metrics? "
        )

    # A dictionary.
    if isinstance(results, dict):
        if objective.name not in results:
            raise errors.FatalValueError(
                f"Expected the returned dictionary from {func_name} to have "
                f"the specified objective, {objective.name}, "
                "as one of the keys. "
                f"Received: {results}."
            )
        return

    # Other unsupported types.
    raise errors.FatalTypeError(
        f"Expected the return value of {func_name} to be "
        "one of float, dict, keras.callbacks.History, "
        "or a list of one of these types. "
        f"Recevied return value: {results} of type {type(results)}."
    )
