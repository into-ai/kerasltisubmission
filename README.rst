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

This python package allows you to submit your trained ``keras`` models to a
``kerasltiprovider`` (see `here <https://github.com/into-ai/kerasltiprovider>`_)
that proxies any
`LTI <http://www.imsglobal.org/activity/learning-tools-interoperability>`_
conforming learning platform for grading.

.. code-block:: console

    $ pip install kerasltisubmission

.. code-block:: python

    import kerasltisubmission

    # Looking for the provider?
    # See https://github.com/into-ai/kerasltiprovider
    provider = LTIProvider(
        input_api_endpoint="http://localhost:8080",
        submission_api_endpoint="http://localhost:8080/submit",
        user_token="7dd7367c-40c2-43cb-a052-bb04e1d0a858",
    )

    # Submit your keras model
    submission = Submission(assignment_id=12, model=model)
    provider.submit(submission)

For a complete example, see `example.py <example.py>`_.
Also see the `official documentation`_ as well as the
`documentation <https://github.com/into-ai/kerasltiprovider>`_ of
the ``kerasltiprovider``.

.. _official documentation: https://kerasltisubmission.readthedocs.io

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
