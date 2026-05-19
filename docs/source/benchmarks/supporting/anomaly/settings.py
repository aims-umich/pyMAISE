"""
settings.py.

Configurations settings for global vairbales.
"""

import pyMAISE as mai
from pathlib import Path

anomaly_dir = Path(__file__).parent

# Data paths — None triggers automatic download from Mendeley and caching
# in the pyMAISE OS cache directory (~/.cache/pyMAISE on Linux/macOS).
# Set to a local file path only if you already have the DTL dataset on disk.
input_path = None
output_path = None

# pyMAISE settings
problem_type = mai.ProblemType.CLASSIFICATION
data_random_state = 42
random_state = None
verbosity = 3

# Data loading settings
non_faulty_frac = 0.3

# Train/test split and rolling windows
test_size = 0.3
