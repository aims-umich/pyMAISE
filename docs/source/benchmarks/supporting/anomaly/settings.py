"""
settings.py.

Configurations settings for global vairbales.
"""

import pyMAISE as mai

# Data paths
input_path = "../pyMAISE/tests/supporting/DTL.npy"
output_path = "../pyMAISE/tests/supporting/DTL_labels.npy"

# pyMAISE settings
problem_type = mai.ProblemType.CLASSIFICATION
random_state = 42
verbosity = 3

# Data loading settings
non_faulty_frac = 0.5

# Train/test split and rolling windows
test_size = 0.3
