Install
=======

Prerequisites
-------------
- PowerWorld Simulator with SimAuto (COM interface) enabled
- Python 3.10+ and ``pip`` available on your path

Install the package
-------------------

Use the latest published package:

.. code-block:: bash

    python -m pip install esapp

For development against this repository:

.. code-block:: bash

    python -m pip install -e .

Verify the installation
-----------------------

.. code-block:: python

    from esapp import GridWorkBench
    wb = GridWorkBench("path/to/your/case.pwb")
    print(wb)

Next steps
----------
- Continue to the :doc:`usage` guide for indexing and API basics
- See :doc:`examples` for end-to-end notebooks
- Review :doc:`../api/api` for full reference
