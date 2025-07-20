import pathlib

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class IntersimDataset(Dataset):
    """InterSIM Dataset class"""

    def __init__(self, data, labels, sample_ids=None, transform=None):
        self.df = torch.tensor(data.values, dtype=torch.float32)
        self.labels = labels
        self.sample_ids = sample_ids
        self.transform = transform
        self.classes = torch.unique(torch.tensor(labels.loc[:,'cluster.id'].values))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        x = self.df[idx,:]
        y = self.labels.loc[self.labels.index[idx],'cluster.id'].item()

        if self.transform:
            x = self.transform(x)

        if self.sample_ids is not None:
            s = self.sample_ids[idx]
            return x, y, s
        
        return x, y

def load_dataset(
    dataset_name,
    pairing: str = 'unpaired',
    split: str = 'train',
    data_dir: str = '../data'
) -> list[pd.DataFrame]:
    """Load the InterSIM generated data"""

    if split not in ['train', 'test']:
        raise Exception(f"Valid dataset splits are 'train' or 'test'")

    if pairing not in ['paired', 'unpaired']:
        raise Exception(f"Valid pairing options are 'paired' and 'unpaired'")
    
    dataset_omics = ['dna_methylation', 'gene_expression', 'protein_expression']
    
    data = []
    for omic in dataset_omics:
        x_path = pathlib.Path(data_dir) / dataset_name / f"{dataset_name}_{omic}_{pairing}_x_{split}.tab"
        y_path = pathlib.Path(data_dir) / dataset_name / f"{dataset_name}_{omic}_{pairing}_y_{split}.tab"
        x_df = pd.read_csv(x_path, sep='\t', header=0, index_col=0)
        y_df = pd.read_csv(y_path, sep='\t', header=0, index_col=0)
        data.append((x_df, y_df))

    return data

def get_datasets(
    dataset_name: str,
    pairing: str = 'unpaired',
    split: str = 'train',
    data_dir: str = './data',
    include_sample_ids: bool = False
) -> list[IntersimDataset]:
    """Load the InterSIM generated dataset and prepare it for loading"""
    
    # Load the InterSIM dataset
    data = load_dataset(
        dataset_name = dataset_name,
        pairing = pairing,
        split = split,
        data_dir = data_dir
    )
    
    # Normalization
    scaler = MinMaxScaler()
    normalized = [
        (pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index), y-1)  # Adjust labels to zero-index
        for x, y in data
    ]
    
    # Prepare the data for loading
    intersim_datasets = [
        IntersimDataset(x, y, sample_ids=y.index if include_sample_ids else None) for x, y in normalized 
    ]

    return intersim_datasets
