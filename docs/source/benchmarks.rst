.. _examples:

===========================
Benchmark Jupyter Notebooks
===========================

Welcome to the pyMAISE benchmarked Jupyter Notebooks! These notebooks include the machine learning benchmarks for reactor control, reactor physics, fuel performance, and heat conduction data sets. Follow the :doc:`benchmarks/mit_reactor` notebook for an introductory tutorial with pyMAISE. The other notebooks focus on the models and methods instead of applying pyMAISE.

**Regression**

.. toctree::
   :maxdepth: 1

   benchmarks/mit_reactor.ipynb
   benchmarks/reactor_physics.ipynb
   benchmarks/fuel_performance.ipynb
   benchmarks/heat_conduction.ipynb
   benchmarks/bwr.ipynb
   benchmarks/HTGR_microreactor.ipynb
   benchmarks/rod_ejection.ipynb
   benchmarks/chf.ipynb

**Classification**

-----------------------
Creating Your Benchmark
-----------------------

pyMAISE aims to be a medium for AI/ML researchers to benchmark their data sets and models with standard ML methods; subsequently, we encourage you to contribute if you are interested. Here, we outline the procedure for creating a pyMAISE benchmark. Please read the :ref:`dev_guide` before continuing.

1. On the pyMAISE GitHub, open a `Benchmark Request <https://github.com/myerspat/pyMAISE/issues/new?assignees=&labels=&projects=&template=benchmark-request.md&title=>`_ under Issues. Please describe the data set you plan to use in the benchmark and link any relevant resources, such as a link to the published paper (if there is one).
2. Perform the following:
    1. Add the data to ``pyMAISE/datasets/``.
    2. Add a load function to ``pyMAISE/datasets/_handler.py`` and include a description of the data. This load function should return ``xarray.DataArray``.
    3. Create and run a Jupyter notebook for the benchmark in ``docs/source/benchmarks/``.
    4. Add the relative path to the notebook to ``docs/source/benchmarks.rst`` under the ``toctree``.
    5. Add a blurb about the data to the user guid under the :ref:`preprocessing <preprocessing>` section in ``docs/souce/user_guide.rst``.
    6. If a published paper exists for the data set, add the BibTeX citation to ``docs/source/data_refs.bib``.
3. Once these steps are completed, you can push the benchmark, ensuring to adhere to the workflow outlined in the :ref:`dev_guide`, and create a `pull request <https://github.com/myerspat/pyMAISE/pulls>`_.

A reviewer will ensure the validity of the benchmark and data. They will offer feedback and possible revisions for you. Thank you for contributing!
