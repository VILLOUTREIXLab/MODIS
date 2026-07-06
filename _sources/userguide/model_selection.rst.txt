.. _model_selection:

Hyperparameter Search
=====================

MODIS provides utilities in :mod:`modis.utils.config` (and mirrored in
:mod:`modis.utils.model_selection`) to systematically search over
hyperparameter grids.

Generating a Grid
-----------------

:func:`~modis.utils.config.generate_grid` takes a base configuration file and
a dictionary mapping parameter names to lists of candidate values.  It returns
one validated :class:`omegaconf.DictConfig` per unique combination:

.. code-block:: python

   from pathlib import Path
   from modis.utils.config import generate_grid

   grid = generate_grid(
       base_config_file=Path("config_base.yaml"),
       params={
           "latent_size":    [32, 64, 128],
           "learning_rate":  [1e-4, 2e-4, 5e-4],
           "beta":           [1e-4, 1e-3, 1e-2],
       },
   )

   print(f"Grid size: {len(grid)} configurations")   # 3 × 3 × 3 = 27

Running the Grid
----------------

Iterate over the grid and call :func:`~modis.train.train` for each
configuration:

.. code-block:: python

   from modis import train
   from pathlib import Path

   results = []
   for i, config in enumerate(grid):
       print(f"\n=== Configuration {i+1}/{len(grid)} ===")
       checkpoint_dir = train(
           config=config,
           train_datasets=train_datasets,
           val_datasets=val_datasets,
           show_dataset_summary=False,
           run_evaluation=True,
           generate_plots=False,
           read_args=False,
       )
       results.append(checkpoint_dir)

Nested Parameter Names
----------------------

Nested configuration fields are accessed using dot notation.  For example,
to vary the ``encoder_ratios`` of the first modality:

.. code-block:: python

   grid = generate_grid(
       base_config_file=Path("config_base.yaml"),
       params={
           "modalities.0.encoder_ratios": [
               [1.0, 0.5, 0.25],
               [1.2, 0.75, 0.5, 0.25],
           ],
           "latent_size": [32, 64],
       },
   )

Setting Parameters Individually
--------------------------------

To update a single parameter on an existing config object:

.. code-block:: python

   from modis.utils.config import set_param

   set_param(config, "latent_size", 128)
   set_param(config, "modalities.0.input_size", 512)

Flattening a Config for Logging
---------------------------------

:func:`~modis.utils.config.flatten_omegaconf` converts a nested config into a
flat dictionary with dot-notation keys, which is convenient for logging to
tools like MLflow or Weights & Biases:

.. code-block:: python

   from modis.utils.config import flatten_omegaconf

   flat = flatten_omegaconf(config)
   # {"dataset_name": "pbmc_cite", "latent_size": 64,
   #  "modalities.0.name": "rna", "modalities.0.input_size": 2000, ...}

   import mlflow
   mlflow.log_params(flat)

Configuration Validation
------------------------

:func:`~modis.utils.config.validate_config` is called automatically by
:func:`~modis.utils.io.load_config`, :func:`~modis.utils.config.config_from_dict`,
and :func:`~modis.utils.config.generate_grid`.  It performs the following
checks:

- All required fields are present.
- ``training_mode`` is ``"supervised"`` or ``"semisupervised"``.
- ``device`` is ``"auto"``, ``"cuda"``, or ``"cpu"``.
- All numeric fields are within their valid ranges.
- ``modalities`` contains at least two entries, each with ``name`` and a
  positive ``input_size``.
- ``num_classes`` is present and positive when ``training_mode == "supervised"``.
- ``beta`` is a float or a list whose length matches the number of modalities.
- ``save_checkpoint_latest`` and ``save_checkpoint_best`` are booleans.

If any check fails, the program exits with an informative error message.