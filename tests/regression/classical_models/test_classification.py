from sklearn.model_selection import ShuffleSplit

import pyMAISE as mai
from pyMAISE.preprocessing import one_hot_encode, read_csv, train_test_split


def test_classification():
    # ===========================================================================
    # pyMAISE initialization
    global_settings = mai.init(
        problem_type=mai.ProblemType.CLASSIFICATION,
        verbosity=1,
        random_state=42,
        num_configs_saved=1,
        cuda_visible_devices="-1",
    )

    # Assertions for global settings
    assert global_settings.verbosity == 1
    assert global_settings.random_state == 42
    assert global_settings.num_configs_saved == 1

    # Get iris data
    data, inputs, outputs = read_csv(
        "https://raw.githubusercontent.com/scikit-learn/scikit-learn/04e39db499"
        + "afab852e4e2603807384a402a871a9/sklearn/datasets/data/iris.csv",
        slice(0, 4),
        slice(4, 5),
    )
    # Change output data type and one hot encode
    outputs = outputs.astype("object")
    outputs = one_hot_encode(outputs)

    # Assert inputs and outputs are the correct size
    assert data.shape == (150, 5)
    assert inputs.shape == (150, 4)
    assert outputs.shape == (150, 3)

    # Train test split
    split_data = train_test_split([inputs, outputs], test_size=0.3)

    # Train-test split size assertions
    assert split_data[0].shape == (105, 4)
    assert split_data[1].shape == (45, 4)
    assert split_data[2].shape == (105, 3)
    assert split_data[3].shape == (45, 3)

    # ===========================================================================
    # Model initialization
    model_settings = {
        "models": ["DT", "RF", "KN"],
    }
    tuning = mai.Tuner(split_data[0], split_data[2], model_settings=model_settings)

    # ===========================================================================
    # Hyper-parameter tuning
    grid_search_spaces = {
        "DT": {
            "max_depth": [None, 5, 10, 25, 50],
            "max_features": [None, "sqrt", "log2"],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
        },
        "RF": {
            "n_estimators": [50, 100, 150],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_split": [2, 4, 6],
            "max_features": ["sqrt", "log2"],
        },
        "KN": {
            "n_neighbors": [1, 2, 4, 6, 8, 10, 14, 17, 20],
            "weights": ["uniform", "distance"],
            "leaf_size": [1, 5, 10, 15, 20, 25, 30],
        },
    }

    grid_search_configs = tuning.grid_search(
        param_spaces=grid_search_spaces,
        models=grid_search_spaces.keys(),
        cv=ShuffleSplit(
            n_splits=1, test_size=0.15, random_state=global_settings.random_state
        ),
    )

    # ===========================================================================
    # Model post-processing
    postprocessor = mai.PostProcessor(
        data=split_data,
        model_configs=[grid_search_configs],
    )

    # Assert accuracy >= 0.9; exact value varies by numpy version due to
    # sort/tie-breaking changes affecting the train/test split
    for key in ["DT", "RF", "KN"]:
        assert (
            postprocessor.metrics(model_type=key)["Test Accuracy"].to_numpy()[0] >= 0.9
        )
