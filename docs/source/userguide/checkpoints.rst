.. _checkpoints:

Checkpoints
===========

MODIS uses PyTorch's ``torch.save`` / ``torch.load`` mechanism to persist
model and optimiser states.  This page describes the checkpoint format, how
to save and load checkpoints, and how to resume interrupted training.

Checkpoint Format
-----------------

Each ``checkpoint_best.pth`` or ``checkpoint_latest.pth`` file is a plain
Python dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``model_state``
     - ``dict`` — PyTorch state dict of the full :class:`~modis.nn.Model`.
   * - ``optimizer_state``
     - ``dict`` — PyTorch state dict of the shared Adam optimiser.
   * - ``epoch``
     - ``int`` — zero-based index of the epoch at which this checkpoint was
       saved.
   * - ``best_epoch``
     - ``int`` — zero-based index of the best epoch seen so far.
   * - ``best_loss``
     - ``float`` — lowest ``g_loss`` value seen so far.
   * - ``val_acc``
     - ``float`` — validation accuracy associated with the best epoch
       (``0.0`` if no validation data were used).
   * - ``timestamp``
     - ``str`` — training-run timestamp in ``YYYYMMDD_HHMMSS`` format; used
       to reconstruct the checkpoint directory path on resume.

Loading a Checkpoint
--------------------

Use :func:`~modis.utils.io.load_checkpoint` to retrieve the raw dict, then
pass it to the model or trainer:

.. code-block:: python

   from modis.utils.io import load_checkpoint
   from pathlib import Path

   checkpoint_data = load_checkpoint(
       Path("./saved/checkpoints/pbmc_cite/run_01/20240501_143022/checkpoint_best.pth")
   )
   print(f"Saved at epoch {checkpoint_data['epoch']}")

Loading Model Weights Only
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only need the model (e.g., for inference):

.. code-block:: python

   from modis import Model, load_config

   config = load_config(checkpoint_dir / "config.yaml")
   model = Model(config)
   model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")
   model.eval()

Loading from a State Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from modis.utils.io import load_checkpoint

   checkpoint_data = load_checkpoint(checkpoint_dir / "checkpoint_best.pth")
   model.load_from_state_dict(checkpoint_data["model_state"])

.. Resuming Training
.. -----------------

.. Training can be resumed from the command line using the ``--checkpoint``
.. argument:

.. .. code-block:: bash

..    python -m modis.train \
..        --checkpoint ./saved/checkpoints/pbmc_cite/run_01/20240501_143022/checkpoint_best.pth

.. MODIS will:

.. 1. Load model and optimiser states from the checkpoint.
.. 2. Reload the training log from ``checkpoint_log_best.json`` (or
..    ``checkpoint_log_latest.json``).
.. 3. Read ``config.yaml`` from the checkpoint directory.
.. 4. Continue training from ``checkpoint['epoch'] + 1``.

.. .. warning::

..    When resuming, the ``config.yaml`` in the checkpoint directory is always
..    used — any configuration file passed programmatically is overridden.

Training Logs
-------------

Each checkpoint is accompanied by a JSON log file containing per-epoch metric
dictionaries.  Use :func:`~modis.utils.io.load_log` to read it:

.. code-block:: python

   from modis.utils.io import load_log
   from pathlib import Path

   log = load_log(
       Path("./saved/checkpoints/pbmc_cite/run_01/20240501_143022/checkpoint_log_best.json")
   )

   # log is a list of dicts, one per epoch
   for entry in log:
       print(entry["epoch_idx"], entry["g_loss"], entry.get("val_acc", "N/A"))

Inspecting a Checkpoint Directory
-----------------------------------

A complete checkpoint directory contains:

.. code-block:: text

   20240501_143022/
   ├── config.yaml                          ← full OmegaConf config
   ├── checkpoint_best.pth                  ← best model state
   ├── checkpoint_log_best.json             ← epoch log for best checkpoint run
   ├── checkpoint_latest.pth                ← latest model state (if enabled)
   ├── checkpoint_log_latest.json           ← epoch log for latest checkpoint run
   └── checkpoints_evaluation_metrics.json  ← post-training evaluation results