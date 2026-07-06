.. _training:

Training
========

MODIS provides two entry points for training:

- :func:`~modis.train.train` — the recommended high-level wrapper that runs
  the training loop, evaluates checkpoints, and generates diagnostic plots.
- :func:`~modis.train.train_loop` — the lower-level epoch loop used internally
  by ``train``; useful when you want fine-grained control over post-training
  steps.

It is recommended to train the model until the reconstruction loss plateaus.


High-Level Training with ``train``
------------------------------------

.. code-block:: python

   from modis import load_config, train

   config = load_config("config.yaml")
   checkpoint_dir = train(
       config=config,
       train_datasets=train_datasets,
       val_datasets=val_datasets,   # optional
       show_dataset_summary=True,
       run_evaluation=True,
       generate_plots=True,
       checkpoint_path=None,
       read_args=False,             # always False when calling programmatically
   )

Arguments
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``config``
     - A validated :class:`omegaconf.DictConfig` loaded with
       :func:`~modis.utils.io.load_config`.
   * - ``train_datasets``
     - List of ``Dataset`` objects, one per modality.
   * - ``val_datasets``
     - List of ``Dataset`` objects for validation.  ``None`` disables
       per-epoch validation accuracy tracking.
   * - ``show_dataset_summary``
     - Prints a tabular summary of dataset sizes before training.
   * - ``run_evaluation``
     - After training, evaluates all saved checkpoints and writes a
       ``checkpoints_evaluation_metrics.json`` file.
   * - ``generate_plots``
     - After training, generates diagnostic plots for each saved checkpoint.
   * - ``checkpoint_path``
     - ``pathlib.Path`` path object to the checkpoint to be resumed or ``None``.
   * - ``read_args``
     - When ``True``, ``argparse`` is used to read ``--checkpoint`` from
       ``sys.argv`` (intended for CLI use).  Set to ``False`` in notebooks
       and scripts.

Return Value
~~~~~~~~~~~~

:func:`~modis.train.train` returns a :class:`pathlib.Path` pointing to the
checkpoint directory, or ``None`` if no checkpoints were saved (i.e., both
``save_checkpoint_best`` and ``save_checkpoint_latest`` are ``false``).

Epoch Metrics
-------------

The following metrics are printed each epoch:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``recon_loss``
     - Mean squared error reconstruction loss summed over all modalities.
   * - ``kl_loss``
     - KL divergence loss (β-weighted) summed over all modalities.
   * - ``d_train_loss``
     - Total discriminator training loss (adversarial + auxiliary + clustering).
   * - ``d_loss``
     - Discriminator feedback included in the generator loss.
   * - ``g_loss``
     - Total generator loss (recon + KL + d_loss).  This is the primary metric
       used for best-checkpoint selection.
   * - ``d_aux_acc``
     - Auxiliary classifier accuracy on labeled samples in the current batch.
   * - ``val_acc``
     - Validation accuracy (only printed when ``val_datasets`` is provided).

.. Command-Line Training
.. ---------------------

.. The training script can also be invoked directly from the command line.
.. The ``--checkpoint`` flag resumes training from a saved checkpoint:

.. .. code-block:: bash

..    python -m modis.train --checkpoint ./saved/checkpoints/my_dataset/modis_v1/20240101_120000/checkpoint_best.pth

.. When ``--checkpoint`` is supplied:

.. - Model and optimiser states are restored.
.. - The training log from the checkpoint run is reloaded.
.. - The original configuration file is loaded from the checkpoint directory
..   (any ``config`` argument passed programmatically is ignored).
.. - Training continues from ``checkpoint['epoch'] + 1``.

Checkpoint Saving Strategy
---------------------------

Two checkpoint variants can be saved, controlled by boolean config flags:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Behaviour
   * - ``save_checkpoint_best: true``
     - Saves ``checkpoint_best.pth`` whenever the current epoch achieves a
       lower ``g_loss`` (and, if validation data are provided, a non-decreasing
       ``val_acc``) compared to all previous epochs.
   * - ``save_checkpoint_latest: true``
     - Saves ``checkpoint_latest.pth`` once at the very end of training,
       capturing the final model state.

Both flags can be ``true`` simultaneously.

Checkpoint Directory Layout
----------------------------

Checkpoints are stored under ``./saved/checkpoints/<dataset_name>/<model_name>/<timestamp>/``:

.. code-block:: text

   saved/
   └── checkpoints/
       └── pbmc_cite/
           └── latent64_beta1e-3/
               └── 20240501_143022/
                   ├── config.yaml
                   ├── checkpoint_best.pth
                   ├── checkpoint_log_best.json
                   ├── checkpoint_latest.pth
                   ├── checkpoint_log_latest.json
                   └── checkpoints_evaluation_metrics.json

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Contents
   * - ``config.yaml``
     - The full OmegaConf configuration used for this run.
   * - ``checkpoint_best.pth`` / ``checkpoint_latest.pth``
     - PyTorch checkpoint dict (model state, optimiser state, metadata).
   * - ``checkpoint_log_best.json`` / ``checkpoint_log_latest.json``
     - Per-epoch metric history as a JSON list of dicts.
   * - ``checkpoints_evaluation_metrics.json``
     - Post-training evaluation results for both checkpoints on train/val data.

Memory and GPU Tips
-------------------

- Reduce ``batch_size`` if you run out of GPU memory.
- Use ``device: cpu`` for debugging on machines without a GPU.
- Call ``del trainer; torch.cuda.empty_cache()`` (done automatically by
  :func:`~modis.train.train_loop`) to free GPU memory after training.
- For very large datasets, reduce ``num_epochs`` and use
  ``save_checkpoint_latest`` to capture the final state.