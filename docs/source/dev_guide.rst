.. _dev_guide:

=========
Dev Guide
=========

-------------
Prerequisites
-------------

pyMAISE currently supports Python 3.9, 3.10, and 3.11.

.. admonition:: Required
   :class: error

   -  `NumPy <https://numpy.org/>`_
   -  `pandas <https://pandas.pydata.org/>`_
   -  `Xarray <https://docs.xarray.dev/en/stable/index.html>`_
   -  `scikit-learn <https://scikit-learn.org/stable/index.html>`_
   -  `scikit-optimize <https://scikit-optimize.github.io/stable/>`_
   -  `Keras <https://keras.io>`_
   -  `KerasTuner <https://keras.io/keras_tuner/>`_
   -  `TensorFlow <https://tensorflow.org>`_
   -  `SciKeras <https://adriangb.com/scikeras/stable/>`_
   -  `Matplotlib <https://matplotlib.org/stable/>`_
   -  `pytest <https://docs.pytest.org/en/8.0.x/>`_
   -  `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/index.html>`_
   -  `pre-commit <https://pre-commit.com/>`_
   -  `Flake8 <https://flake8.pycqa.org/en/latest/>`_
   -  `Black <https://black.readthedocs.io/en/stable/index.html>`_
   -  `docformatter <https://docformatter.readthedocs.io/en/latest/>`_
   -  `Sphinx <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
   -  `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_
   -  `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_
   -  `Jupyter <https://jupyter.org/>`_
   -  `OpenCV <https://opencv.org/>`_
   -  `SciPy <https://scipy.org/>`_

----------------------
Installation and Setup
----------------------

Before cloning the repository, git and Python must be installed on your Linux distribution. You can do this through your Linux package manager. On Ubuntu/Debian, run ``sudo apt-get install git``. Once git is installed, `generate an SSH key and add it to GitHub <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux>`_. With git setup, the repository can be cloned and installed into the directory of your choice with the following commands

.. code-block:: sh

   # Clone the repository
   git clone git@github.com:aims-umich/pyMAISE.git
   cd pyMAISE/

   # Checkout the develop branch
   git checkout develop

   # Install pyMAISE
   pip install -e ".[dev]"

---------
Branching
---------

In the previous section, we checked out the development branch. This branch is the repository's main branch and is never directly edited. Before writing your code, create a new branch. Branches are always made off of development, so before any new branch, ensure that your development branch is up to date.

.. code-block:: sh

   # Checkout the develop branch
   git checkout develop

   # Get the latest version of the develop branch from GitHub
   git pull

   # Create your new working branch off of develop called `branch-name`
   git checkout -b branch-name

Before each branch, update your latest develop version with ``git pull``. Additionally, the ``branch-name`` can be anything you'd like and is preferably a name related to the changes/issue the branch is for. Now, you can edit the repository code on your new branch. To keep your branch up to date with develop, run ``git pull origin develop``. A new branch should be made for each issue as a best practice.

----------
Committing
----------

The code must be committed and pushed to see any changes you make to the source code reflected on the develop branch on GitHub. Before this, follow the :ref:`precommit` section to set up the pyMAISE pre-commit hook. Committing entails staging and then committing your staged changes with a short message describing the changes you made.

.. code-block:: sh

   # Stage the changed file for committing
   git add path/to/file

   # Commit the changes with a short descriptive message
   git commit -m "what I changed"

Commit often and write strong messages so reviewers can easily understand what was changed and why.

-------
Pushing
-------

Changes committed can now be pushed, assuming they pass all tests and the code runs without issues. To make your branch to GitHub, run ``git push -u origin branch-name``; this will set an upstream link to the remote branch on the server so further changes can be pushed with just ``git push``.

.. _precommit:

-----------------------
Install Pre-commit Hook
-----------------------

To enforce programming standards and formatting across pyMAISE, we include a pre-commit hook that runs Black, docformatter, and Flake8 before each commit. pyMAISE uses Black and docformatter for formatting, and Flake8 is a Python linter that enforces PEP 8 standards. To install the pre-commit hook run

.. code-block:: sh

   pre-commit install

The pre-commit hook only checks these standards and does not automatically reformat code. If any of these checks fail, the commit is stopped. To format a file, run ``black <source_file_or_directory>`` and ``docformatter -i <source_file_or_directory>``.

----------------
General Workflow
----------------

Changes should be made only if there is a representative issue in the issue tab of the GitHub repository with detailed information on what should change and why. The problem can then be assigned to a contributor, a branch can be made, and coding can begin. Once the branch is ready, it can be pushed to the remote repository, and a pull request (PR) can be made for that branch to be pulled into development. The PR should outline what changes were made, why, and what issue the PR closes. The PR must then be reviewed by someone other than the original contributor. The branch may be pulled into development if the code passes all tests and the reviewer is happy with the work. The reviewer may request changes, and you should make the changes and push them.

-------
Testing
-------

Run the following to run the pyMAISE regression and unit test suite:

.. code-block:: sh

   pytest

Run the tests before each push. These tests are also run within the continuous integration in GitHub Actions with each push to a pull request, testing Python 3.9, 3.10, and 3.11.
