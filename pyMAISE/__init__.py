# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

from pyMAISE.postprocessor import PostProcessor
from pyMAISE.settings import ProblemType, init
from pyMAISE.tuner import Tuner
from pyMAISE.utils import Boolean, Choice, Fixed, Float, Int, _try_clear

_try_clear()

# This should always be the last line of this file
__version__ = "1.0.0b0"
