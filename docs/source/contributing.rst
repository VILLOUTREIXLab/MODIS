.. _contributing:

Contributing
============

We welcome contributions! This page outlines how to set up a development
environment, and submit changes.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/your-org/modis.git
   cd modis
   pip install -e .

Code Style
----------

MODIS follows `PEP 8 <https://peps.python.org/pep-0008/>`_ with a maximum
line length of 100 characters.  Docstrings use
`Google style <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.


Building the Docs
-----------------

.. code-block:: bash

   cd docs
   make html
   open build/html/index.html

Submitting a Pull Request
--------------------------

1. Fork the repository and create a feature branch.
2. Make your changes.
3. Update the relevant documentation pages.
4. Open a pull request describing your changes.