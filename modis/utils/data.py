import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class PartiallyLabeledDataset(Dataset):
    """Adjust the labels in a labeled dataset for semisupervised mode with MODIS"""

    def __init__(
        self,
        dataset,
        labeled_ratio: float = None,
        labeled_samples: int = None,
        class_samples: int | list[int|None] = None,
        class_samples_ratio: float | list[float|None] = None,
        num_random_samples: int = None,
        remove_unlabeled: bool| list[int] = False,
        random_seed: int | None = None
    ) -> None:
        """
        Specify the amount of samples that keep their labels (the rest of the labels are turned to -1)

        Args:
            labeled_ratio: Set a ratio of random labeled to unlabeled samples (from 0 to 1) (randomly distributed among classes)
            labeled_samples: Number of random labeled vs unlabeled samples (randomly distributed among classes)
            class_samples: Fixed number of random labeled vs unlabeled samples per class if integer,
                           if list (of the same length as the number of classes in the dataset)
                           specify the number of random labeled samples per class
            class_samples_ratio: Same as class_samples but for a ratio of labeled samples (from 0 to 1)

            num_random_samples: This dataset will have a subset of the given dataset with this number of samples,
                                the previous parameter are applied to these samples instead of the complete dataset 
            remove_unlabeled: If True, unlabeled samples are removed, if list, unlabeled samples from the classes in this list are removed
            random_seed: For reproducibility
        """
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
            random.seed(random_seed)
        else:
            rng = np.random.default_rng()

        self.dataset = dataset

        if num_random_samples is not None:
            assert num_random_samples <= len(dataset), f"dataset only has {len(dataset)} samples"
            random_dataset = rng.choice(range(len(self.dataset)), size=num_random_samples, replace=False)  ### requires seed??
            self.dataset = Subset(dataset, random_dataset)

        total_samples = len(self.dataset)
        all_indices = rng.permutation(total_samples)

        if labeled_ratio is not None:
            num_labeled = int(total_samples * labeled_ratio)
            self.labeled_indices = set(all_indices[:num_labeled])

        elif labeled_samples is not None:
            self.labeled_indices = set(all_indices[:labeled_samples])

        elif class_samples is not None or class_samples_ratio is not None:
            class_indices = dict() # index of samples per class
            for i, (_, label) in enumerate(self.dataset):
                if not label in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            self.labeled_indices = []
            for label in class_indices:
                if class_samples is not None:
                    if type(class_samples) == list:
                        num_labeled = class_samples[label] if class_samples[label] is not None else len(class_indices[label])
                    else:
                        num_labeled = class_samples
                else:
                    if type(class_samples_ratio) == list:
                        num_labeled = int(len(class_indices[label]) * class_samples_ratio[label]) if class_samples_ratio[label] is not None else len(class_indices[label])
                    else:
                        num_labeled = int(len(class_indices[label]) * class_samples_ratio)
                self.labeled_indices.extend(random.sample(class_indices[label], num_labeled))
        else:
            self.labeled_indices = set(all_indices)

        if type(remove_unlabeled) == list:
            keep = self.labeled_indices  + [i for i, (_, label) in enumerate(self.dataset) if i not in self.labeled_indices and label not in remove_unlabeled]
            self.dataset = Subset(self.dataset, keep)
            total_samples = len(self.dataset)
            self.labeled_indices = set(range(len(self.labeled_indices)))
        elif remove_unlabeled == True:
            self.dataset = Subset(self.dataset, self.labeled_indices)
            total_samples = len(self.dataset)
            self.labeled_indices = set(range(total_samples))

        # Index of labeled samples
        self.is_labeled = [i in self.labeled_indices for i in range(total_samples)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.is_labeled[index]:
            return x, y
        else:
            # Adjust sample label to -1 (to represent unlabeled)
            return x, -1


def get_dataloaders(
    datasets: list[torch.utils.data.Dataset],
    batch_size: int,
    drop_last: bool = True,
    shuffle: bool = True
) -> list[torch.utils.data.DataLoader]:
    """Return a dataloader for each dataset"""
    dataloaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
                   for ds in datasets]
    return dataloaders

def summarize_dataset(dataloaders: list[torch.utils.data.DataLoader], modality_names: list | None = None) -> None:
    """
    Print the total number of samples in a dataloader and their class distribution.
    
    Note: Requires labeled datasets
    """
    if modality_names:
        assert len(modality_names) == len(dataloaders), "The dataloaders and modality names provided should have the same length"

    if modality_names is None:
        modality_names = [i for i in range(len(dataloaders))]

    total_samples = Counter()
    for i, dataloader in enumerate(dataloaders):
        assert isinstance(dataloader, torch.utils.data.DataLoader), f"You must provide a list of Pytorch DataLoaders"

        dataset = dataloader.dataset
        class_counter = Counter([data[1] for data in dataset])
        total_samples = total_samples + class_counter

        sorted_class_counter = dict(sorted(class_counter.items()))
        # print(f"Dataset {modality_names[i]} ({sum(sorted_class_counter.values())} samples), samples per class: {sorted_class_counter}")
        print(f"Dataset {modality_names[i]} ({sum(sorted_class_counter.values())} samples)")
        print(f"Samples per class: {sorted_class_counter}")
        print()

    sorted_total_samples = dict(sorted(total_samples.items()))
    print(f"Total samples: {sum(sorted_total_samples.values())}: {sorted_total_samples}")

    if -1 in sorted_total_samples:
        num_labeled = sum([sorted_total_samples[label] for label in sorted_total_samples if label != -1])
        print(f"Global labeled samples ratio: {round(num_labeled / sum(sorted_total_samples.values()), 3)}")

def get_samples_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = None,
    device: str = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read a specific number of samples from a dataloader
    """
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    if num_samples == None:
        num_samples = dataset_size

    if num_samples > dataset_size:
        raise Exception(f"The dataset only has {dataset_size} samples")

    samples = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            x, y = data[0], data[1]
            if device is not None:
                x, y = x.to(device), y.to(device)
            samples.append(x)
            labels.append(y)
            if len(samples)*batch_size >= num_samples:
                break

    samples = torch.cat(samples, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]

    return samples, labels