Installation
============


Create a new virtual environment
--------------------------------

You can create a vitual environment as follows:

.. code-block:: bash

    mkdir modis
    cd modis
    python -m venv env

To activate the environment:

.. code-block:: bash

    source env/bin/activate


Install from Source
-------------------

With the environment activated, install MODIS using `pip`.

.. code-block:: bash

   git clone https://github.com/VILLOUTREIXLab/MODIS.git
   cd MODIS
   pip install .
