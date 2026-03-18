Data inputs
===========

Passing your data to MODIS is straightforward and very flexible.
You can load your datasets using any method you prefer — for example, with libraries such as **AnnData**, **Polars**, or **Pandas**.  
Each modality must then be wrapped in a ``torch.utils.data.Dataset`` class.
All modality-specific ``Dataset`` instances are finally passed to MODIS as a list of inputs.

Example Dataset
---------------

Consider a dataset with two modalities: **DNA methylation** and **mRNA expression**, both stored as ``.csv`` (or ``.tab``) files.

.. table:: Modality 1: Methylation
   :align: left

   +---------+---------+---------+
   | Sample  | CpG1    | CpG2    |
   +=========+=========+=========+
   | Sample1 | 0.82    | 0.67    |
   +---------+---------+---------+
   | Sample2 | 0.75    | 0.70    |
   +---------+---------+---------+
   | Sample3 | 0.80    | 0.65    |
   +---------+---------+---------+


.. table:: Modality 1: Methylation class labels
   :align: left

   +---------+-------+
   | Sample  | label |
   +=========+=======+
   | Sample1 | 0     |
   +---------+-------+
   | Sample2 | 3     |
   +---------+-------+
   | Sample3 | 1     |
   +---------+-------+

.. table:: Modality 2: Gene Expression
   :align: left

   +---------+-------+-------+-------+
   | Sample  | GeneA | GeneB | GeneC |
   +=========+=======+=======+=======+
   | Sample1 |  5.2  |  3.1  |  7.0  |
   +---------+-------+-------+-------+
   | Sample2 |  4.8  |  3.5  |  6.8  |
   +---------+-------+-------+-------+
   | Sample3 |  5.0  |  3.2  |  7.2  |
   +---------+-------+-------+-------+

.. table:: Modality 2: Gene Expression class labels
   :align: left

   +---------+-------+
   | Sample  | label |
   +=========+=======+
   | Sample1 | -1    |
   +---------+-------+
   | Sample2 | 0     |
   +---------+-------+
   | Sample3 | 2     |
   +---------+-------+

Standardizing the file names can help manage the different modalities as you can see in the next section. A pattern for a file name may be: `<modality>_<pairing>_<data(x)/labels(y)>_<split>.tab`, for example:   `dna_methylation_unpaired_x_train.tab`.


Loading the Data
----------------

Below is an example implementation using **Pandas** to load data and **PyTorch** to define a simple ``Dataset`` class for each modality.

Each dataset object wraps the data and its corresponding labels and can optionally include the sample identifiers that can be useful to track samples across modalities later.

.. code-block:: python

    from pathlib import Path

    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset

    class ModalityDataset(Dataset):

        def __init__(self, data_path, labels_path, include_ids=False, transform=None):
            self.data = pd.read_csv(data_path, sep='\t', index_col=0)
            self.labels = pd.read_csv(labels_path, sep='\t', index_col=0)
            self.include_ids = include_ids
            self.transform = transform
            
            # Convert to numpy for faster indexing
            self.data_values = self.data.values.astype(np.float32)
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            x = torch.from_numpy(self.data_values[idx])
            y = torch.tensor(self.labels.iloc[idx]['label'], dtype=torch.long)
            
            if self.transform:
                x = self.transform(x)
                
            if self.include_ids:
                return x, y, self.data.index[idx]
            return x, y

    def get_datasets(dataset_name: str, split: str, include_ids=False, data_dir='.'):
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split '{split}'. Valid options are 'train' or 'test'.")

        data_dir = Path(data_dir) / dataset_name
        modalities = ['dna_methylation', 'gene_expression']
        
        datasets = []
        for modality in modalities:
            data_path = data_dir / f"{modality}_unpaired_x_{split}.tab"
            labels_path = data_dir / f"{modality}_unpaired_y_{split}.tab"
            datasets.append(ModalityDataset(data_path, labels_path, include_ids))
        
        return datasets


Code Description
----------------

- **ModalityDataset**  
  Defines how each modality (e.g., methylation, expression) is represented for PyTorch training.  
  Each instance loads:
  - a feature matrix (``x``)
  - a label vector (``y``)
  - optionally, the sample identifiers (``Sample1``, ``Sample2``, etc.)

- **get_datasets**  
  Automatically constructs one ``ModalityDataset`` per modality for a given data split (train or test).  
  The returned list can be directly passed to MODIS for training or evaluation.

  This function assumes your files follow the standard naming convention described above.


Important Notes
---------------

- It is **strongly recommended** to standardize and normalize your datasets before using them with MODIS.
- All class labels must be **zero-based** (i.e., start at ``0``).
- Samples with unknown or missing labels must be assigned a value of ``-1``.
- MODIS can handle partially labeled data, but at least a minimal number of labeled samples per class per modality is required for proper alignment across modalities.


``Tip``:
You can easily adapt this code to:
- Add new modalities (e.g., proteomics, metabolomics).  
- Use data transformations such as normalization or feature selection.  
- Switch to other I/O backends (e.g., **Polars** or **AnnData**) for large-scale datasets.
