"""Adapt this code according to your own datasets"""
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ModalityDataset(Dataset):

    def __init__(self, data_path, labels_path, include_ids=False, transform=None):
        self.data = pd.read_csv(data_path, sep='\t', index_col=0)
        self.labels = pd.read_csv(labels_path, sep='\t', index_col=0)
        self.include_ids = include_ids
        self.transform = transform

        # Standardize the data
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(normalized_data, columns=self.data.columns, index=self.data.index)
        
        # Adjust labels to zero-indexed
        self.labels['label'] = self.labels['label'] - 1
        
        # Convert to numpy for faster indexing
        self.data_values = self.data.values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_values[idx])
        y = torch.tensor(self.labels.iloc[idx]['label'], dtype=torch.long) #.tolist()

        if self.transform:
            x = self.transform(x)
            
        if self.include_ids:
            return x, y, self.data.index[idx]
        return x, y

def get_datasets(dataset_name: str, split: str, include_ids=False, data_dir='.'):
    if split not in ['train', 'test']:
        raise ValueError(f"Invalid split '{split}'. Valid options are 'train' or 'test'.")

    data_dir = Path(data_dir) / dataset_name
    modalities = ['dna_methylation', 'gene_expression', 'protein_expression']
    
    datasets = []
    for modality in modalities:
        data_path = data_dir / f"{modality}_unpaired_x_{split}.tab"
        labels_path = data_dir / f"{modality}_unpaired_y_{split}.tab"
        datasets.append(ModalityDataset(data_path, labels_path, include_ids))
    
    return datasets