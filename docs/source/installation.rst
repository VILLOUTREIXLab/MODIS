.. _installation:

Installation
============


Installing from Source
----------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/your-org/modis.git
   cd modis
   pip install -e .

Installing Dependencies Only
----------------------------

If you have cloned the repository but only want to install the runtime
dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Verifying the Installation
--------------------------

Open a Python interpreter and run:

.. code-block:: python

   import modis
   print(modis.__version__)

GPU Support
-----------

MODIS automatically detects CUDA when ``device: auto`` is set in the
configuration (see :ref:`configuration`). To use a specific device, set
``device: cuda`` or ``device: cpu`` explicitly.

To verify GPU availability:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))  # if a GPU is present