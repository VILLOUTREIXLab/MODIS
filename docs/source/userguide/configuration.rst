.. _configuration:

Configuration
=============

MODIS uses `OmegaConf <https://omegaconf.readthedocs.io/>`_ YAML files for
all experiment settings.  Configurations are loaded and validated with
:func:`~modis.utils.io.load_config`, which calls
:func:`~modis.utils.config.validate_config` internally.

You can also build a configuration from a plain Python dictionary using
:func:`~modis.utils.config.config_from_dict`.

Loading a Configuration
-----------------------

.. code-block:: python

   from modis import load_config

   config = load_config("path/to/config.yaml")

From a dictionary:

.. code-block:: python

   from modis.utils.config import config_from_dict

   config = config_from_dict({
       "dataset_name": "my_dataset",
       "model_name": "run_01",
       "latent_size": 64,
       # ...
   })

Reference: All Fields
---------------------

The following table lists every field that can appear in a configuration file.
Fields marked **required** must always be present.

.. list-table::
   :header-rows: 1
   :widths: 25 12 10 53

   * - Field
     - Type
     - Required
     - Description
   * - ``dataset_name``
     - ``str``
     - ✓
     - Identifier for the dataset.  Used as part of the checkpoint directory
       path.
   * - ``model_name``
     - ``str``
     - ✓
     - Identifier for the model / experiment run.
   * - ``latent_size``
     - ``int``
     - ✓
     - Dimensionality of the shared latent space.  Must be > 0.
   * - ``num_classes``
     - ``int``
     - ✓
     - Number of output classes for the auxiliary classifier head.  Must
       be > 0.
   * - ``modalities``
     - ``list``
     - ✓
     - List of modality configuration blocks (see `Modality Fields`_ below).
       At least two modalities are required.
   * - ``training_mode``
     - ``str``
     - ✓
     - ``"supervised"`` or ``"semisupervised"``.
   * - ``batch_size``
     - ``int``
     - ✓
     - Mini-batch size.  Must be > 0.
   * - ``num_epochs``
     - ``int``
     - ✓
     - Total number of training epochs.  Must be > 0.
   * - ``learning_rate``
     - ``float``
     - ✓
     - Learning rate for the Adam optimiser (shared by VAEs and
       Discriminator).  Must be > 0.
   * - ``beta``
     - ``float`` or ``list[float]``
     - ✓
     - KL divergence weight.  A scalar applies the same weight to all
       modalities; a list must have the same length as ``modalities``.
   * - ``beta1``
     - ``float``
     - ✓
     - Adam β₁ parameter.  Must be in [0, 1].
   * - ``lambda_r``
     - ``float``
     - ✓
     - Weight for the zero-centred gradient penalty (R1 regularisation).
       Must be ≥ 1.
   * - ``device``
     - ``str``
     - ✓
     - ``"auto"``, ``"cuda"``, or ``"cpu"``.  ``"auto"`` selects CUDA
       when available, otherwise falls back to CPU.
   * - ``save_checkpoint_best``
     - ``bool``
     - ✓
     - If ``true``, saves a checkpoint whenever the model achieves a new
       best generator loss (and, if validation data are provided, a
       non-decreasing validation accuracy).
   * - ``save_checkpoint_latest``
     - ``bool``
     - ✓
     - If ``true``, saves a ``checkpoint_latest.pth`` at the end of
       training regardless of performance.

Modality Fields
---------------

Each entry in the ``modalities`` list is a sub-configuration with the
following fields:

.. list-table::
   :header-rows: 1
   :widths: 25 12 10 53

   * - Field
     - Type
     - Required
     - Description
   * - ``name``
     - ``str``
     - ✓
     - Human-readable name for the modality (e.g., ``"rna"``, ``"atac"``).
   * - ``input_size``
     - ``int``
     - ✓
     - Number of input features for this modality.  Must be > 0.
   * - ``encoder_ratios``
     - ``list[float]``
     - ✗
     - Ratios used to compute hidden layer sizes relative to ``input_size``.
       The decoder uses the reversed list.  Defaults to
       ``[1.2, 1.0, 0.75, 0.5, 0.25]`` when omitted.  All values must be
       > 0.

Example Configuration
---------------------

.. code-block:: yaml

   dataset_name: pbmc_cite
   model_name: latent64_beta1e-3

   modalities:
     - name: rna
       input_size: 2000
       encoder_ratios: [1.0, 0.75, 0.5, 0.25]
     - name: protein
       input_size: 134
       encoder_ratios: [1.5, 1.0, 0.5]

   latent_size: 64
   num_classes: 14

   training_mode: semisupervised
   batch_size: 256
   num_epochs: 200
   learning_rate: 0.0002
   beta: [0.001, 0.005]   # per-modality KL weights
   beta1: 0.5
   lambda_r: 10.0

   device: auto

   save_checkpoint_best: true
   save_checkpoint_latest: true

Programmatic Configuration
---------------------------

For hyperparameter searches, MODIS provides utilities to generate a grid of
configurations programmatically.  See :ref:`model_selection` for details.