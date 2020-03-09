===============================
kerasltisubmission
===============================

.. image:: https://travis-ci.com/into-ai/kerasltisubmission.svg?branch=master
        :target: https://travis-ci.com/into-ai/kerasltisubmission
        :alt: Build Status

.. image:: https://img.shields.io/pypi/v/kerasltisubmission.svg
        :target: https://pypi.python.org/pypi/kerasltisubmission
        :alt: PyPI version

.. image:: https://img.shields.io/github/license/into-ai/kerasltisubmission
        :target: https://github.com/into-ai/kerasltisubmission
        :alt: License

.. image:: https://readthedocs.org/projects/kerasltisubmission/badge/?version=latest
        :target: https://kerasltisubmission.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/into-ai/kerasltisubmission/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/into-ai/kerasltisubmission
        :alt: Test Coverage

""""""""

Your short description here. `into-ai.github.io/kerasltisubmission <https://into-ai.github.io/kerasltisubmission>`_

.. code-block:: console

    $ pip install kerasltisubmission

See the `official documentation`_ for more information.

.. _official documentation: https://kerasltisubmission.readthedocs.io

.. code-block:: python

    import kerasltisubmission

Development
-----------

For detailed instructions see `CONTRIBUTING <CONTRIBUTING.rst>`_.

Tests
~~~~~~~
You can run tests with

.. code-block:: console

    $ invoke test
    $ invoke test --min-coverage=90     # Fail when code coverage is below 90%
    $ invoke type-check                 # Run mypy type checks

Linting and formatting
~~~~~~~~~~~~~~~~~~~~~~~~
Lint and format the code with

.. code-block:: console

    $ invoke format
    $ invoke lint

All of this happens when you run ``invoke pre-commit``.

Note
-----

This project is still in the alpha stage and should not be considered production ready.
