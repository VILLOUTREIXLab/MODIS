import random
from itertools import combinations
from typing import Any
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class PartiallyLabeledDataset(Dataset):
    """Adjust the labels in a labeled dataset for semisupervised mode with MODIS"""

    def __init__(
        self,
        dataset,
        labeled_samples_ratio: float = None,
        labeled_samples: int = None,
        labeled_class_samples: int | list[int|None] = None,
        labeled_class_samples_ratio: float | list[float|None] = None,
        num_random_samples: int = None,
        remove_unlabeled: bool| list[int] = False,
        random_seed: int | None = None
    ) -> None:
        """
        Adjust the number of samples that retain their labels (remaining
        labels are set to -1).

        Args:
            labeled_samples_ratio (float): 
                Ratio (0–1) of labeled samples randomly distributed across
                classes relative to unlabeled samples in the dataset.
            labeled_samples (int):
                Number of samples to retain labels, randomly distributed
                across classes.
            labeled_class_samples (int | list):
                Fixed number of labeled samples per class. If an integer,
                applies to all classes; if a list (matching the number of
                classes), specifies labeled samples per class.
            labeled_class_samples_ratio (float | list):
                Same as labeled_class_samples, but specifies the ratio (0 to 1)
                of labeled samples per class.
            num_random_samples (int):
                Generate a subset with this number of samples; previous 
                parameters apply to this subset.
            remove_unlabeled (bool): 
                If True, all unlabeled samples are removed; if a list,
                unlabeled samples from the specified classes are removed.
            random_seed (int):
                Integer seed to ensure reproducible random operations
                across runs.
        """
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
            random.seed(random_seed)
        else:
            rng = np.random.default_rng()

        self.dataset = dataset

        if num_random_samples is not None:
            assert num_random_samples <= len(dataset), f"Dataset only has {len(dataset)} samples"
            random_dataset = rng.choice(range(len(self.dataset)), size=num_random_samples, replace=False)  ### requires seed??
            self.dataset = Subset(dataset, random_dataset)

        total_samples = len(self.dataset)
        all_indices = rng.permutation(total_samples)

        if labeled_samples_ratio is not None:
            num_labeled = int(total_samples * labeled_samples_ratio)
            self.labeled_indices = set(all_indices[:num_labeled])

        elif labeled_samples is not None:
            self.labeled_indices = set(all_indices[:labeled_samples])

        elif labeled_class_samples is not None or labeled_class_samples_ratio is not None:
            class_indices = dict()  # Index of samples per class
            for i, (_, label) in enumerate(self.dataset):
                if not label in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            self.labeled_indices = []
            for label in class_indices:
                if labeled_class_samples is not None:
                    if type(labeled_class_samples) == list:
                        num_labeled = labeled_class_samples[label] if labeled_class_samples[label] is not None else len(class_indices[label])
                    else:
                        num_labeled = labeled_class_samples
                else:
                    if type(labeled_class_samples_ratio) == list:
                        num_labeled = int(len(class_indices[label]) * labeled_class_samples_ratio[label]) if labeled_class_samples_ratio[label] is not None else len(class_indices[label])
                    else:
                        num_labeled = int(len(class_indices[label]) * labeled_class_samples_ratio)
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

def random_split(
        dataset: Dataset | list[Dataset], 
        length_ratios: list[float],
        paired: bool,
        random_seed: int | None = None
) -> list:
    """
    Randomly split a dataset or multi-modal dataset
    
    Args:
        dataset: Single Dataset or list of Datasets
        length_ratios: List of ratios for each split (should sum to 1.0)
        random_seed: Random seed for reproducibility
        paired: If True, ensures corresponding samples across datasets stay together
    
    Returns:
        List of splits, where each split contains Dataset(s) in the same format as input
    """
    # Handle single dataset case
    if isinstance(dataset, Dataset):
        return torch.utils.data.random_split(
            dataset=dataset,
            lengths=length_ratios,
            generator=torch.Generator().manual_seed(random_seed) if random_seed is not None else None
        )
    
    # Multi-modal dataset case
    if not dataset:
        raise ValueError("Dataset list cannot be empty")
    
    # Verify all datasets have the same length for paired datasets
    dataset_length = len(dataset[0])
    if paired and not all(len(ds) == dataset_length for ds in dataset):
        raise ValueError("All datasets must have the same length for paired splitting")
    
    if not paired:
        # Split each dataset independently
        splits = list(zip(*[torch.utils.data.random_split(
            dataset=ds,
            lengths=length_ratios,
            generator=torch.Generator().manual_seed(random_seed) if random_seed is not None else None
        ) for ds in dataset]))
        return splits
    
    # Paired dataset splitting - generate indices once and apply to all datasets
    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    indices = torch.randperm(dataset_length, generator=generator).tolist()
    
    # Convert ratios to actual lengths

    if abs(sum(length_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Length ratios must sum to 1.0, got {sum(length_ratios)}")
    
    lengths = []
    remaining = dataset_length
    for ratio in length_ratios[:-1]:
        length = int(ratio * dataset_length)
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)  # Last split gets remaining samples
    
    # Create index splits
    splits_indices = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        splits_indices.append(indices[start_idx:end_idx])
        start_idx = end_idx
    
    # Create Subset datasets for each split and each modality
    splits = []
    for split_indices in splits_indices:
        split_datasets = [Subset(ds, split_indices) for ds in dataset]
        splits.append(split_datasets)
    
    return splits

def _list_of_combinations(n: int) -> list[tuple[int, ...]]:
    """
    Generates all possible combinations of n items.

    Args:
        n: The total number of items.

    Returns:
        A list of tuples, where each tuple contains a unique combination (from 0 to n-1).
    """
    indices = list(range(n))
    return [comb for r in range(1, n + 1) for comb in combinations(indices, r)]

def _find_common_intersection(lists: list[list[Any]]) -> list[Any]:
    """
    Finds the common intersection among sublists.

    Args:
        lists (list[list]): List of lists of items to be compared.

    Returns:
        A list containing the elements that are present in all sublists.
        Returns an empty list if there is no intersection.
    """
    if not lists:
        return []
    return list(set.intersection(*map(set, lists)))

def _unique_intersections(indexes_list: list[list[str | int]]) -> dict[tuple[int, ...], list[str | int]]:
    """
    Return the unique intersections across all combinations of the input sublists.

    Args:
        indexes_list (list[list[str | int]]): 
            A list of sublists, where each sublist contains identifiers 
            (e.g., strings or integers) corresponding to a given modality or group.

    Returns:
        dict[tuple[int, ...], list[str | int]]:
            A dictionary mapping:
              - Keys: Tuples of indices indicating which sublists (modalities) 
                form the combination.
              - Values: Sorted lists of items that are present in *all* sublists 
                of the given combination, but absent from every other sublist 
                outside the combination.
    """
    num_modalities = len(indexes_list)
    combination_dict = {}

    # Iterate through all possible combinations of modalities
    for comb_tuple in _list_of_combinations(num_modalities):
        # Convert the tuple of combination indices to a list for easier indexing
        comb_list = list(comb_tuple)

        # Get the lists of sample IDs for the current combination of modalities
        current_modalities_indexes = [indexes_list[i] for i in comb_list]

        # Calculate the intersection of sample IDs for the current combination
        intersection = set(_find_common_intersection(current_modalities_indexes))

        # Identify the indices of the modalities *not* in the current combination
        other_modalities_indices = set(range(num_modalities)) - set(comb_list)
        other_indexes = set()

        # Collect all sample IDs from the *other* modalities
        for i in other_modalities_indices:
            other_indexes.update(indexes_list[i])

        # Find the sample IDs that are unique to the intersection of the current
        # combination (i.e., present in the intersection but not in any other modality)
        combination_dict[comb_tuple] = sorted(list(intersection - other_indexes))

    return combination_dict

def multimodal_dataset_split(
    indexes_list: list[list[str | int]],
    test_ratio: float = 0.2, 
    stratify_by: list[list[int]] = None,
    paired_only: bool = False,
    random_seed: int = None
) -> tuple[dict[int, list[Any]], dict[int, list[Any]]]:
    """
    Generate train/test splits for multi-modal datasets.

    Args:
        indexes_list (list[list[str | int]]): 
            A list of sublists, where each sublist contains sample identifiers 
            corresponding to a given modality or group.
        test_ratio: The proportion of samples to allocate to the test set (default: 0.2).

    Returns:
        A tuple containing two dictionaries:
        - The first dictionary (`train_samples_per_modality`) has modality indices as keys
          and lists of training sample IDs for that modality as values.
        - The second dictionary (`test_samples_per_modality`) has modality indices as keys
          and lists of testing sample IDs for that modality as values.
    """
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")

    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)

    # Initialize dictionaries to store training and testing samples per modality
    num_modalities = len(indexes_list)
    train_samples_per_modality = {i: [] for i in range(num_modalities)}
    test_samples_per_modality = {i: [] for i in range(num_modalities)}

    # Get the dictionary of unique intersections per combination of modalities
    intersections_dict = _unique_intersections(indexes_list)

    if stratify_by is not None:
        if len(stratify_by) != num_modalities:
            raise ValueError("stratify_by must have the same length as indexes_list")
        for i in range(len(stratify_by)):
            if len(stratify_by[i]) != len(indexes_list[i]):
                raise ValueError(f"stratify_by[{i}] must have the same length as indexes_list[{i}]")
        # sample_id to class mapping
        sample_id_class_dict = {k: v for row_k, row_v in zip(indexes_list, stratify_by) for k, v in zip(row_k, row_v)}

    # Iterate through each combination of modalities and its unique intersecting samples
    for comb_tuple, unique_samples in intersections_dict.items():
        if paired_only and len(comb_tuple) != num_modalities: continue
        
        if not unique_samples:
            continue  # Skip if there are no unique samples for this combination

        n_total = len(unique_samples)
        n_test = round(n_total * test_ratio)

        # Ensure there are enough samples for the test set
        if n_test <= 0 and n_total > 0:
            raise ValueError(f"Not enough unique samples ({n_total}) for combination {comb_tuple} to create a test set with ratio {test_ratio}.")

        if stratify_by is None:
            # Randomly sample test samples from the unique samples
            test_samples = random.sample(unique_samples, n_test)
            # The remaining unique samples form the training set for this combination
            train_samples = list(set(unique_samples) - set(test_samples))
        else:
            unique_samples_class = [sample_id_class_dict[i] for i in unique_samples]
            classes_dict = {c:[] for c in set(unique_samples_class)}
        
            for sidx, s in enumerate(unique_samples):
                c = unique_samples_class[sidx]
                classes_dict[c].append(s)

            class_nsamples_dict = {k:round(n_test * len(v)/n_total) for k,v in classes_dict.items()}
           
            test_samples = []
            for ic, n_samples in class_nsamples_dict.items():
                # if class_nsamples_dict[ic] <= 0:
                #     raise ValueError(f"Not enough unique samples ({n_total}) for combination {comb_tuple} to create a test set with ratio {test_ratio}")
                test_samples.extend(random.sample(classes_dict[ic], n_samples))
            test_samples = sorted(test_samples)

            train_samples = list(set(unique_samples) - set(test_samples))
        
        # Add the training and testing samples to the respective dictionaries
        # for each modality involved in the current combination
        for i_mod in comb_tuple:
            train_samples_per_modality[i_mod].extend(train_samples)
            test_samples_per_modality[i_mod].extend(test_samples)

    for k,v in train_samples_per_modality.items():
        train_samples_per_modality[k] = sorted(v)

    for k,v in test_samples_per_modality.items():
        test_samples_per_modality[k] = sorted(v)

    return train_samples_per_modality, test_samples_per_modality

def stratified_k_fold(k: int, datasets: list[torch.utils.data.Dataset], sample_ids: list[str|int] | None, random_seed: int | None = None) -> list[list[str]]:
    """
    Generator that returns stratified k folds of train and test datasets for a multi-modal dataset

    Args:
        k (int): Number of folds to divide each dataset into
        datastes (list[torch.utils.data.Dataset]): Input dataset
        sample_ids (list[str|int] | None): List of sample ids for paired or partially paired datasets
                                           Use None if unpaired
        random_seed (int): Seed for reproducibility
    """
    assert k >= 2, "K-fold cross-validation requires k >= 2"

    if sample_ids is None:
        # Assume unpaired samples
        sample_ids = [
            [
                sample_index + sum([len(ds) for i,ds in enumerate(datasets) if i<mi])
                for sample_index,_ in enumerate(ds)
            ] 
            for mi,ds in enumerate(datasets)
        ]
    else:
        sample_ids = sample_ids.copy()

    classes = [[sample[1] for sample in ds] for ds in datasets]
    idtoidx = [{sid:i for i,sid in enumerate(modality_ids)} for modality_ids in sample_ids]

    # Generate folds
    folds = []
    for n in range(k, 0, -1):
        _, test_indexes = multimodal_dataset_split(
            sample_ids,
            test_ratio=1/n,
            stratify_by=classes,
            paired_only=False,
            random_seed=random_seed
        )
        folds.append(test_indexes)

        # Remove test_indexes from sample_ids and classes for next loop
        for modality_index in range(len(sample_ids)):
            indexes_to_remove = [i for i,idx in enumerate(sample_ids[modality_index]) if idx not in test_indexes[modality_index]]

            sample_ids[modality_index] = [idx for i,idx in enumerate(sample_ids[modality_index]) if i in indexes_to_remove]
            classes[modality_index] = [sample_class for i,sample_class in enumerate(classes[modality_index]) if i in indexes_to_remove]
    
    # Generate train, test datasets from k-folds
    for test_index in range(k):
        test_fold_ids = folds[test_index]
        train_folds = folds[:test_index] + folds[test_index+1:]

        # Concatenate train folds
        train_fold = defaultdict(list)
        for fold in train_folds:
            for key, value in fold.items():
                train_fold[key].extend(value)        
        train_fold_ids = dict(train_fold)

        # Convert sample ids to sample indexes
        test_fold_indexes = {
            modality_id: [idtoidx[modality_id][sample_id] for sample_id in modality]
            for modality_id, modality in test_fold_ids.items()
        }

        train_fold_indexes = {
            modality_id: [idtoidx[modality_id][sample_id] for sample_id in modality]
            for modality_id, modality in train_fold_ids.items()
        }
        
        # Generate train, test dataset for this k-fold
        test_ds = [Subset(ds, test_fold_indexes[di]) for di, ds in enumerate(datasets)]
        train_ds = [Subset(ds, train_fold_indexes[di]) for di, ds in enumerate(datasets)]
        
        yield train_ds, test_ds
