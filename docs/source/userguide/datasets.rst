.. _datasets:

Preparing Datasets
==================

MODIS expects one ``torch.utils.data.Dataset`` per modality.  Each dataset
must return ``(features, label)`` tuples where:

- ``features`` is a 1-D ``torch.Tensor`` of shape ``(input_size,)``.
- ``label`` is an integer scalar ``torch.Tensor``.  Unlabeled samples should
  carry a label of ``-1``.

Dataset Contract
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Property
     - Requirement
   * - ``__len__``
     - Must return the number of samples.
   * - ``__getitem__``
     - Must return a ``(features_tensor, label_tensor)`` tuple.
   * - Label ``-1``
     - Treated as *unlabeled* in semi-supervised mode.  Raises an exception in
       supervised mode.
   * - Paired samples
     - Datasets across modalities should contain **the same number of
       samples**, and samples at the same index should correspond to the same
       biological or physical object.

Using ``TensorDataset``
-----------------------

The simplest approach is ``torch.utils.data.TensorDataset``:

.. code-block:: python

   import torch
   from torch.utils.data import TensorDataset

   N = 500        # number of samples
   d_A, d_B = 128, 256

   X_A = torch.randn(N, d_A)   # modality A features
   X_B = torch.randn(N, d_B)   # modality B features
   labels = torch.randint(0, 5, (N,))

   # Mark the first 400 samples as unlabeled
   labels[:400] = -1

   train_datasets = [
       TensorDataset(X_A, labels),
       TensorDataset(X_B, labels),
   ]

Writing a Custom Dataset
------------------------

For more complex data loading (e.g., reading from HDF5 or CSV files), subclass
``torch.utils.data.Dataset``:

.. code-block:: python

   import torch
   from torch.utils.data import Dataset

   class MyModalityDataset(Dataset):
       def __init__(self, features, labels):
           # features: numpy array or tensor of shape (N, input_size)
           # labels:   array of int, -1 for unlabeled
           self.features = torch.as_tensor(features, dtype=torch.float32)
           self.labels   = torch.as_tensor(labels,   dtype=torch.long)

       def __len__(self):
           return len(self.labels)

       def __getitem__(self, idx):
           return self.features[idx], self.labels[idx]

Validation Datasets
-------------------

A separate list of validation datasets can be passed to
:func:`~modis.train.train` and :func:`~modis.train.train_loop`:

.. code-block:: python

   val_datasets = [
       TensorDataset(X_A_val, labels_val),
       TensorDataset(X_B_val, labels_val),
   ]

   from modis import train
   checkpoint_dir = train(config, train_datasets, val_datasets=val_datasets)

When validation datasets are provided:

- A :func:`~modis.utils.evaluation.evaluate_model` call is made after each
  epoch to compute validation accuracy.
- The best-checkpoint criterion becomes
  ``g_loss < best_loss AND val_acc >= prev_best_val_acc``.
- Evaluation reports are generated for both training and validation sets after
  training finishes.

.. note::

   Validation datasets should contain **only labeled** samples (no ``-1``
   labels), as the evaluation metrics are computed over labeled data only.

Data Normalisation
------------------

MODIS does not apply any data normalisation internally.  It is strongly
recommended to standardise or normalise your features before training:

.. code-block:: python

   from sklearn.preprocessing import StandardScaler

   scaler_A = StandardScaler()
   X_A_scaled = torch.tensor(
       scaler_A.fit_transform(X_A.numpy()), dtype=torch.float32
   )