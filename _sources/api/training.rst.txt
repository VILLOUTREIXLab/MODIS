.. _api_training:

Training (``modis.training``)
==============================

This section documents the training orchestration layer: the
:class:`~modis.training.Trainer` class, the epoch-based training loop
functions, and the loss functions used during training.

Top-Level Functions
-------------------

These functions are importable directly from ``modis``:

.. autofunction:: modis.train.train

.. autofunction:: modis.train.train_loop

.. autofunction:: modis.train.parse_args

Trainer
-------

.. automodule:: modis.training.trainer
   :no-members:

.. autoclass:: modis.training.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Loss Functions (``modis.training.losses``)
------------------------------------------

.. automodule:: modis.training.losses
   :no-members:

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   DDCLoss
   EntropyLoss
   ClusteringLoss

.. autoclass:: modis.training.losses.DDCLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: modis.training.losses.EntropyLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: modis.training.losses.ClusteringLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__