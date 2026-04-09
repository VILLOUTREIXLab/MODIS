.. _quickstart:

Quickstart
==========

This page walks through a minimal end-to-end example: defining a
configuration, preparing datasets, training a model, and evaluating it.

Step 1 — Define a Configuration
--------------------------------

Create a YAML file (e.g., ``config.yaml``) describing your experiment:

.. code-block:: yaml

   # config.yaml
   dataset_name: my_dataset
   model_name: modis_v1

   # Modalities
   modalities:
     - name: view_A
       input_size: 128
       encoder_ratios: [1.0, 0.75, 0.5, 0.25]
     - name: view_B
       input_size: 256
       encoder_ratios: [1.2, 1.0, 0.5, 0.25]

   # Latent space
   latent_size: 64
   num_classes: 10

   # Training
   training_mode: semisupervised   # or "supervised"
   batch_size: 128
   num_epochs: 100
   learning_rate: 0.0002
   beta: 0.001
   beta1: 0.5
   lambda_r: 10.0

   # Device
   device: auto   # "auto", "cuda", or "cpu"

   # Checkpointing
   save_checkpoint_best: true
   save_checkpoint_latest: false

Load and validate it with :func:`~modis.utils.io.load_config`:

.. code-block:: python

   from modis import load_config

   config = load_config("config.yaml")

Step 2 — Prepare Datasets
--------------------------

MODIS expects standard ``torch.utils.data.Dataset`` objects that return
``(features, label)`` pairs. Use label ``-1`` for unlabeled samples in
semi-supervised mode.

.. code-block:: python

   from torch.utils.data import TensorDataset
   import torch

   # Synthetic two-modality data — replace with your real datasets
   N, d_A, d_B = 1000, 128, 256

   X_A = torch.randn(N, d_A)
   X_B = torch.randn(N, d_B)
   labels = torch.randint(0, 10, (N,))

   # Mark 80 % of samples as unlabeled for semi-supervised training
   labels[:800] = -1

   train_datasets = [
       TensorDataset(X_A, labels),
       TensorDataset(X_B, labels),
   ]

Step 3 — Train
--------------

Pass the config and datasets to :func:`~modis.train`:

.. code-block:: python

   from modis import train

   checkpoint_dir = train(
       config=config,
       train_datasets=train_datasets,
       val_datasets=None,      # optional; see the User Guide
       run_evaluation=True,
       generate_plots=True,
       read_args=False,        # False when calling programmatically
   )

   print(f"Checkpoints saved to: {checkpoint_dir}")

Training prints per-epoch metrics to stdout:

.. code-block:: text

   epoch: 1/100, recon_loss: 0.8432, kl_loss: 0.0021, d_train_loss: 0.6931,
   d_loss: 0.7102, g_loss: 1.5556, d_aux_acc: 0.1032

Step 4 — Load and Predict
--------------------------

After training, load the best checkpoint and generate predictions:

.. code-block:: python

   from modis import Model, load_config
   import torch

   config = load_config(checkpoint_dir / "config.yaml")
   model = Model(config)
   model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")

   # Predict from modality A
   x_new = torch.randn(32, 128).to(config.device)
   predictions = model.predict(x_new, input_modality=0)
   print(predictions)  # tensor of shape (32,)

Step 5 — Cross-Modal Translation
---------------------------------

Translate samples from modality A into modality B's feature space:

.. code-block:: python

   translated = model.translate(x_new, input_modality=0, output_modality=1)
   print(translated.shape)  # (32, 256)

Next Steps
----------

- :ref:`configuration` — full description of every configuration field.
- :ref:`training` — advanced training options (validation, resuming, CLI).
- :ref:`model_selection` — hyperparameter grid search.
- :ref:`api_nn` — complete API reference for the neural network classes.