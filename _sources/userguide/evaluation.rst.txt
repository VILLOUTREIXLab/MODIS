.. _evaluation:

Evaluation
==========

MODIS provides utilities to evaluate trained models on held-out datasets,
compute a comprehensive set of classification metrics, and inspect
reconstruction quality.

Evaluating a Model
------------------

Use :func:`~modis.utils.evaluation.evaluate_model` to evaluate a loaded model
on one or more DataLoaders:

.. code-block:: python

   from modis import Model, load_config
   from modis.utils.evaluation import evaluate_model
   from modis.utils.data import get_dataloaders

   # Random synthetic two-modality data — replace with your real validation datasets
   N, d_A, d_B = 200, 128, 256

   X_A = torch.randn(N, d_A)
   X_B = torch.randn(N, d_B)
   labels = torch.randint(0, 10, (N,))

   val_datasets = [
    TensorDataset(X_A, labels),
    TensorDataset(X_B, labels),
   ]

   config = load_config(checkpoint_dir / "config.yaml")
   model = Model(config)
   model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")

   # Wrap datasets in DataLoaders
   dataloaders = get_dataloaders(
       val_datasets,
       batch_size=config.batch_size,
       drop_last=False,
       shuffle=False,
   )

   metrics = evaluate_model(model, dataloaders)
   for name, value in metrics.items():
       print(f"{name}: {value}")

.. note::

   Only samples whose label is ≠ ``-1`` contribute to classification metrics.
   Reconstruction MSE is computed over all samples regardless of label.

Returned Metrics
----------------

:func:`~modis.utils.evaluation.evaluate_model` returns a dict with the
following keys (empty dict if no labeled samples are found):

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Key
     - Type
     - Description
   * - ``acc``
     - ``float``
     - Overall accuracy — fraction of correctly predicted labeled samples.
   * - ``bacc``
     - ``float``
     - Balanced accuracy (macro-averaged per-class recall).  More informative
       for imbalanced datasets.
   * - ``nmi``
     - ``float``
     - Normalized Mutual Information between predicted and true labels.
       Ranges from 0 (no mutual information) to 1 (perfect correspondence).
   * - ``ji``
     - ``float``
     - Macro-averaged Jaccard Index.
   * - ``ari``
     - ``float``
     - Adjusted Rand Index.  Values near 0 indicate random labeling; 1
       indicates perfect agreement.
   * - ``f1``
     - ``float``
     - Weighted-average F1-score.
   * - ``mse``
     - ``float``
     - Mean reconstruction MSE averaged over all modalities and batches.
   * - ``modal_mse``
     - ``list[float]``
     - Per-modality mean reconstruction MSE.

Computing Individual Metrics
----------------------------

Classification metrics can be computed directly on label arrays:

.. code-block:: python

   from modis.utils.evaluation import calc_classification_metrics

   metrics = calc_classification_metrics(
       true_labels=[0, 1, 2, 0, 1],
       pred_labels=[0, 1, 1, 0, 2],
   )
   print(metrics)

Top-1 accuracy from logit tensors:

.. code-block:: python

   import torch
   from modis.utils.evaluation import accuracy

   logits = torch.randn(32, 10)
   targets = torch.randint(0, 10, (32,))
   acc = accuracy(logits, targets)

Batch Checkpoint Evaluation
----------------------------

After training, :func:`~modis.train.train` automatically calls
:func:`~modis.utils.evaluation.launch_checkpoints_evaluation`, which:

1. Loads the config from the checkpoint directory.
2. Evaluates the ``best`` checkpoint (if saved).
3. Evaluates the ``latest`` checkpoint (if saved).
4. Writes results to ``checkpoints_evaluation_metrics.json``.

You can also run this manually:

.. code-block:: python

   from modis.utils.evaluation import launch_checkpoints_evaluation
   from pathlib import Path

   launch_checkpoints_evaluation(
       train_datasets=train_datasets,
       val_datasets=val_datasets,
       checkpoint_dir=Path("./saved/checkpoints/pbmc_cite/run_01/20240501_143022"),
   )

Cross-Modal Pairwise MSE
------------------------

To measure how well latents translate across modalities,
:func:`~modis.utils.evaluation.avg_mse_all_pairs` computes the mean squared
error between every pair of vectors in two tensors:

.. code-block:: python

   import torch
   from modis.utils.evaluation import avg_mse_all_pairs

   X      = torch.randn(100, 64)   # original latents
   X_hat  = torch.randn(100, 64)   # reconstructed latents
   mse    = avg_mse_all_pairs(X, X_hat)
   print(f"Average pairwise MSE: {mse:.6f}")

This is useful for quantifying modality alignment in the latent space.