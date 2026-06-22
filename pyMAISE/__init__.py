import os
import shutil
import sys

# Graphviz 'dot' binary: conda installs it to sys.prefix/bin but doesn't add it
# to PATH when the env is activated from within a notebook kernel.
if shutil.which("dot") is None:
    _gv_bin = os.path.join(sys.prefix, "bin")
    if os.path.isfile(os.path.join(_gv_bin, "dot")):
        os.environ["PATH"] = _gv_bin + os.pathsep + os.environ.get("PATH", "")
    del _gv_bin

# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

from pyMAISE.postprocessor import PostProcessor
from pyMAISE.settings import ProblemType, init
from pyMAISE.tuner import Tuner
from pyMAISE.utils import (
    Boolean,
    Choice,
    Fixed,
    Float,
    Int,
    _try_clear,
    load_tuning_results,
    save_tuning_results,
)
from pyMAISE.explain import _explain as explain
from pyMAISE.explain._explain import ShapExplainers

_try_clear()

# This should always be the last line of this file
__version__ = "1.0.0b0"
