from pyMAISE import IS_NOTEBOOK

# Import output clearing function if notebook
if IS_NOTEBOOK:
    from IPython.display import clear_output


def _try_clear(wait=False):
    """Clear output if pyMAISE is running on a notebook."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    if IS_NOTEBOOK:
        try:
            from pyMAISE.settings import values

            if values.verbosity == 0:
                clear_output(wait)

        except (NameError, ImportError):
            clear_output(wait)
