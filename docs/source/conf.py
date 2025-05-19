# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

project = "pyMAISE"
copyright = "2023, Patrick Myers, Connor Craig, Veda Joynt, Majdi Radaideh"
author = "Patrick Myers, Connor Craig, Veda Joynt, Majdi Radaideh"
release = "1.0.0b0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_default_flags = ["members"]
autosummary_imported_members = True
bibtex_bibfiles = ["software_refs.bib", "data_refs.bib"]
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "_images/pyMAISElogo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "pyMAISE Documentation"
html_favicon = "_images/favicon.ico"
