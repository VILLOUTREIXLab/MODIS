Training a model
================

The following are examples on how to train and evaluate MODIS models.

Basic Supervised Training
--------------------------

The simplest way to train a MODIS model is with fully labeled data:

.. code-block:: python

    import modis
    from src.datasets import get_datasets

    # Load the dataset
    train_datasets = get_datasets(
        dataset_name='toy_dataset',
        split='train',
        include_ids=False,
        data_dir='./data'
    )

    # Train model
    config = modis.load_config('config/supervised.yaml')
    modis.train(config=config, train_datasets=train_datasets)

This approach requires:

A dataset loaded via get_datasets() (a user-constructed function as detailed in :doc:data_inputs)

* A dataset loaded via ``get_datasets()``, a user constructed function as detailed in the :doc:data_inputs :doc:data_inputs <data_inputs> section.
* A configuration file (e.g., ``config/supervised.yaml``)
* Calling ``modis.train()`` with the config and datasets

Semi-Supervised Training
-------------------------

When you have limited labeled data, semi-supervised learning can improve performance:

.. code-block:: python

    import modis
    from modis.utils.data import PartiallyLabeledDataset, random_split
    from src.datasets import get_datasets

    # Load the dataset
    train_datasets = get_datasets(
        dataset_name='toy_dataset',
        split='train',
        include_ids=False,
        data_dir='./data'
    )

    # Generate a validation set
    train_datasets, val_datasets = random_split(
        train_datasets, 
        [0.8, 0.2], 
        paired=False
    )

    # Generate artificially a partially labeled dataset (only 10% supervision)
    train_datasets = [
        PartiallyLabeledDataset(dataset, labeled_samples_ratio=0.10)
        for dataset in train_datasets
    ]

    # Train model
    config = modis.load_config('config/semisupervised.yaml')
    modis.train(
        config=config,
        train_datasets=train_datasets,
        val_datasets=val_datasets
    )

Key features:

* **Validation split**: Use ``random_split()`` to create train/validation sets
* **Partial labeling**: ``PartiallyLabeledDataset`` simulates scenarios with limited labels
* **Semi-supervised config**: Use ``config/semisupervised.yaml`` for appropriate training settings

Complete Training and Evaluation Pipeline
------------------------------------------

For a full workflow including training, checkpointing, and evaluation:

.. code-block:: python

    import modis
    from modis.utils.data import PartiallyLabeledDataset, get_dataloaders, random_split
    from modis.utils.evaluation import evaluate_model
    from src.datasets import get_datasets

    random_seed = 1234
    config = modis.load_config('config/semisupervised.yaml')

    # Load the dataset
    train_datasets = get_datasets(
        dataset_name='toy_dataset',
        split='train',
        include_ids=False,
        data_dir='./data'
    )

    # Generate train/validation data splits
    train_datasets, val_datasets = random_split(
        train_datasets, 
        [0.8, 0.2], 
        random_seed
    )

    # Generate artificially a partially labeled dataset
    train_datasets = [
        PartiallyLabeledDataset(
            dataset, 
            labeled_samples_ratio=0.05, 
            random_seed=random_seed
        )
        for dataset in train_datasets
    ]

    # Train model
    checkpoint_dir = modis.train(
        config=config,
        train_datasets=train_datasets,
        val_datasets=val_datasets
    )

    # Evaluation on test dataset
    print(f"\n==> Evaluating model on test dataset")
    test_datasets = get_datasets(
        dataset_name='toy_dataset',
        split='test',
        include_ids=False,
        data_dir='./data'
    )

    model = modis.Model(config)
    model.load_from_checkpoint(checkpoint_dir / "checkpoint_best.pth")
    
    test_dataloaders = get_dataloaders(
        test_datasets, 
        batch_size=config.batch_size, 
        drop_last=False, 
        shuffle=False
    )
    
    metrics = evaluate_model(model, test_dataloaders)
    
    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"{k}: {[f'{val:.4f}' for val in v]}")
        else:
            print(f"{k}: {v:.4f}")

This pipeline demonstrates:

* **Reproducibility**: Setting ``random_seed`` for consistent splits
* **Checkpoint management**: ``modis.train()`` returns the checkpoint directory
* **Model loading**: Use ``load_from_checkpoint()`` to restore trained models
* **Evaluation**: ``evaluate_model()`` computes metrics on test data

Inference and Translation
--------------------------

After training, you can use the model for prediction, cross-modal translation, and latent extraction:

.. code-block:: python

    import modis
    import modis.utils
    import modis.utils.data
    from src.datasets import get_datasets

    # Prepare the data
    config = modis.load_config('config/semisupervised.yaml')
    checkpoint_file = "saved/checkpoints/<dataset_name>/<model_name>/<timestamp>/checkpoint_best.pth"
    
    datasets = get_datasets(
        dataset_name='toy_dataset',
        split='test',
        include_ids=False,
        data_dir='./data'
    )
    
    dataloaders = modis.utils.data.get_dataloaders(
        datasets,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False
    )

    # Extract samples from each modality
    x, y = list(zip(*[
        modis.utils.data.get_samples_from_dataloader(
            dataloader, 
            num_samples=50, 
            device='cpu'
        )
        for dataloader in dataloaders
    ]))
    
    num_modalities = len(dataloaders)

    # Load checkpoint
    model = modis.Model(config).cpu()
    model.load_from_checkpoint(checkpoint_file)

    # Predict
    pred = [
        model.predict(x[i], input_modality=i) 
        for i in range(num_modalities)
    ]
    print(f"Modality predictions: {[p.numpy().shape for p in pred]}")

    # Translate between modalities
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i == j: 
                continue
            translation = model.translate(
                x[i], 
                input_modality=i, 
                output_modality=j
            )
            print(f"Shape for translation from modality {i} to {j}: {translation.shape}")

    # Extract latent representations
    latents = model.get_latents(x[0], input_modality=0)
    print(f"Latents shape: {latents.numpy().shape}")

Available operations:

* **predict()**: Generate predictions from a specific modality
* **translate()**: Convert data from one modality to another
* **get_latents()**: Extract shared latent representations

Key Parameters
--------------

Dataset Loading
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``dataset_name``
     - Name of the dataset to load
   * - ``split``
     - Dataset split: ``'train'``, ``'test'``, or ``'val'``
   * - ``include_ids``
     - Whether to include sample IDs (default: ``False``)
   * - ``data_dir``
     - Directory containing the data files

Data Splitting
~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``ratios``
     - List of split ratios (e.g., ``[0.8, 0.2]`` for 80/20 split)
   * - ``random_seed``
     - Seed for reproducible splits
   * - ``paired``
     - Whether to maintain pairing across modalities (default: ``False``)

Partial Labeling
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``labeled_samples_ratio``
     - Fraction of samples with labels (e.g., ``0.10`` for 10%)
   * - ``random_seed``
     - Seed for reproducible label selection

Tips and Best Practices
------------------------

1. **Configuration files**: Use separate configs for supervised (``supervised.yaml``) and semi-supervised (``semisupervised.yaml``) training
2. **Random seeds**: Always set ``random_seed`` for reproducible experiments
3. **Validation data**: Include validation sets to monitor training progress and prevent overfitting
4. **Checkpoint paths**: The training function returns the checkpoint directory; use this for evaluation
5. **Batch size**: Configure batch size in your YAML config file based on available memory
6. **Device management**: Move models to appropriate device (``cpu()`` or ``cuda()``) before inference