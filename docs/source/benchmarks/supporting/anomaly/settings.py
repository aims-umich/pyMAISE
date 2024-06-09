"""
settings.py.

Configurations settings for global vairbales.
"""

import pyMAISE as mai
from pathlib import Path

anomaly_dir = Path(__file__).parent

# Data paths (absolute path or relative path from anomaly directory)
input_path = anomaly_dir / "relative/path/to/input.npy"
output_path = anomaly_dir / "relative/path/to/output.npy"

# pyMAISE settings
problem_type = mai.ProblemType.CLASSIFICATION
data_random_state = 42
random_state = None
verbosity = 3

# Data loading settings
non_faulty_frac = 0.3

# Train/test split and rolling windows
test_size = 0.3
