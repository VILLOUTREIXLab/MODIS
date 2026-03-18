Configuration File
==================

MODIS requires a configuration file to specify dataset details, model architecture, training parameters, and checkpoint settings.  
We recommend creating a dedicated directory for all configuration files. All files must be saved in **YAML** format.

Below is an example configuration, along with explanations for each field.

Example
-------

.. code-block:: yaml

    dataset_name: toy_dataset
    model_name: toy01
    latent_size: 64
    modalities:
      - name: dna_methyl
        input_size: 367
      - name: gene_expr
        input_size: 131
      - name: protein_exp
        input_size: 160
        encoder_ratios: [1.2, 1.0]
    training_mode: supervised
    num_classes: 5
    batch_size: 32
    num_epochs: 30
    learning_rate: 1e-4
    beta: 1e-4
    beta1: 0.5
    lambda_r: 10.0
    device: auto
    save_checkpoint_latest: true
    save_checkpoint_best: true


General Settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Key**
     - **Description**

   * - ``dataset_name`` (``str``)
     - Name of the dataset to be used. For compatibility with the data-loading script, this must match the name of an existing dataset directory inside the ``data`` directory.
   * - ``model_name`` (``str``)
     - Identifier for the model that will be trained.


Model architecture
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Key**
     - **Description**

   * - ``latent_size`` (``int``)
     - Dimensionality of the latent space representation for the model.

   * - ``modalities``
     - List of input modalities. Each modality must specify:

       - ``name`` (``str``): Modality identifier (e.g., ``dna_methyl``, ``gene_expr``).
       - ``input_size`` (``int``): Number of features for that modality.
       - ``encoder_ratios`` (``list[float]``): Optional parameter to customize the hidden layers of the modal VAE encoder. Each value in the list specifies the relative size of a hidden layer, expressed as a fraction of the corresponding ``input_size``. Values must be between 0 and 1. The decoder follows the same pattern in inverted order.


Training Parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Key**
     - **Description**

   * - ``training_mode`` (``str``)
     - Training strategy to use. Can be ``supervised`` or ``semisupervised``.

   * - ``num_classes`` (``int``)
     - Number of output classes for the classifier.

   * - ``batch_size`` (``int``)
     - Number of samples per training batch.

   * - ``num_epochs`` (``int``)
     - Total number of training epochs.

   * - ``learning_rate`` (``float``)
     - Optimizer learning rate.

   * - ``beta`` (``float``)
     - Regularization weight, the KL-divergence coefficient for VAEs.

   * - ``beta1`` (``float``)
     - Adam parameter for the optimizer.

   * - ``lambda_r`` (``float``)
     - Weight for reconstruction loss.

   * - ``device`` (``str``)
     - Device to use for training: ``auto``, ``cpu`` or ``cuda``. ``auto`` will automatically select `cuda` if available, `cpu` otherwise.

Checkpoints
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - **Key**
     - **Description**

   * - ``save_checkpoint_latest`` (``bool``)
     - If ``true``, saves a checkpoint at the end of the final epoch.

   * - ``save_checkpoint_best`` (``bool``)
     - If ``true``, saves the model with the best performance on a validation set, or on the training set if validation is unavailable.


Notes
-----

- Ensure that each modality's ``input_size`` matches the corresponding dataset.
- The order of modalities must match the list passed to the training function.
- ``device: auto`` automatically selects GPU if available, otherwise CPU.
- For partially labeled datasets, MODIS can still train models if a minimal number of labeled samples is present.
