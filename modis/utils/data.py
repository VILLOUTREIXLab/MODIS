"""
Dataset and dataloader utilities for MODIS.

This module provides dataset wrappers, dataloader factories, and splitting
helpers for single-modal and multi-modal datasets used in MODIS training and
evaluation.
"""
import random
from itertools import combinations
from typing import Any
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader


class PartiallyLabeledDataset(Dataset):
    """Dataset wrapper that restricts how many samples retain their labels.

    Wraps an existing labeled dataset and sets a configurable subset of
    labels to ``-1`` to simulate a semi-supervised scenario. Exactly one
    of the ``labeled_*`` arguments should be provided.

    Args:
        dataset (torch.utils.data.Dataset): The source labeled dataset.
        labeled_samples_ratio (float, optional): Fraction of samples (0–1)
            that retain their labels, distributed uniformly at random.
        labeled_samples (int, optional): Absolute number of samples that
            retain their labels, chosen uniformly at random.
        labeled_class_samples (int or list[int or None], optional): Fixed
            number of labeled samples per class. Pass an integer to apply the
            same count to all classes, or a list (one entry per class) to
            specify per-class counts. ``None`` entries keep all samples for
            that class.
        labeled_class_samples_ratio (float or list[float or None], optional):
            Same as ``labeled_class_samples`` but expressed as a fraction of
            each class's total sample count.
        num_random_samples (int, optional): Draw a random subset of this size
            from the dataset before applying any labeling constraints.
        remove_unlabeled (bool or list[int], optional): If ``True``, drops all
            unlabeled samples from the dataset. If a list of class indices is
            given, only unlabeled samples belonging to those classes are
            removed. Defaults to ``False``.
        random_seed (int, optional): Seed for the random number generator to
            ensure reproducibility. Defaults to ``None``.
    """

    def __init__(
        self,
        dataset,
        labeled_samples_ratio: float = None,
        labeled_samples: int = None,
        labeled_class_samples=None,
        labeled_class_samples_ratio=None,
        num_random_samples: int = None,
        remove_unlabeled=False,
        random_seed: int = None,
    ) -> None:
        if random_seed is not None:
            rng = np.random.default_rng(seed=random_seed)
            random.seed(random_seed)
        else:
            rng = np.random.default_rng()

        self.dataset = dataset

        if num_random_samples is not None:
            if num_random_samples > len(dataset):
                raise ValueError(
                    f"num_random_samples ({num_random_samples}) exceeds "
                    f"dataset size ({len(dataset)})"
                )
            random_dataset = rng.choice(range(len(self.dataset)), size=num_random_samples, replace=False)
            self.dataset = Subset(dataset, random_dataset)

        total_samples = len(self.dataset)
        all_indices = rng.permutation(total_samples)

        if labeled_samples_ratio is not None:
            num_labeled = int(total_samples * labeled_samples_ratio)
            self.labeled_indices = set(all_indices[:num_labeled].tolist())

        elif labeled_samples is not None:
            self.labeled_indices = set(all_indices[:labeled_samples].tolist())

        elif labeled_class_samples is not None or labeled_class_samples_ratio is not None:
            class_indices = dict()
            for i, (_, label) in enumerate(self.dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            labeled_list = []
            for label in class_indices:
                if labeled_class_samples is not None:
                    if isinstance(labeled_class_samples, list):
                        num_labeled = (
                            labeled_class_samples[label]
                            if labeled_class_samples[label] is not None
                            else len(class_indices[label])
                        )
                    else:
                        num_labeled = labeled_class_samples
                else:
                    if isinstance(labeled_class_samples_ratio, list):
                        num_labeled = (
                            int(len(class_indices[label]) * labeled_class_samples_ratio[label])
                            if labeled_class_samples_ratio[label] is not None
                            else len(class_indices[label])
                        )
                    else:
                        num_labeled = int(len(class_indices[label]) * labeled_class_samples_ratio)
                labeled_list.extend(random.sample(class_indices[label], num_labeled))
            self.labeled_indices = set(labeled_list)
        else:
            self.labeled_indices = set(all_indices.tolist())

        if isinstance(remove_unlabeled, list):
            keep = list(self.labeled_indices) + [
                i for i, (_, label) in enumerate(self.dataset)
                if i not in self.labeled_indices and label not in remove_unlabeled
            ]
            self.dataset = Subset(self.dataset, keep)
            total_samples = len(self.dataset)
            self.labeled_indices = set(range(len(self.labeled_indices)))
        elif remove_unlabeled is True:
            self.dataset = Subset(self.dataset, list(self.labeled_indices))
            total_samples = len(self.dataset)
            self.labeled_indices = set(range(total_samples))

        self.is_labeled = [i in self.labeled_indices for i in range(total_samples)]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if self.is_labeled[index]:
            sample = (
                (sample[0], int(sample[1].item())) + sample[2:]
                if isinstance(sample[1], torch.Tensor)
                else sample
            )
            return sample
        return (sample[0], -1) + sample[2:]


def get_dataloaders(
    datasets: list,
    batch_size: int,
    drop_last: bool = True,
    shuffle: bool = True,
) -> list:
    """Create one DataLoader per dataset.

    Args:
        datasets (list[torch.utils.data.Dataset]): Source datasets.
        batch_size (int): Number of samples per batch.
        drop_last (bool): If ``True``, the last incomplete batch is dropped.
            Defaults to ``True``.
        shuffle (bool): If ``True``, data is shuffled at the start of each
            epoch. Defaults to ``True``.

    Returns:
        list[torch.utils.data.DataLoader]: One DataLoader for each input
        dataset, in the same order.
    """
    dataloaders = [
        torch.utils.data.DataLoader(
            ds, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle
        )
        for ds in datasets
    ]
    return dataloaders


def _print_summary(stats: dict):
    """Helper to format the dictionary output for the console."""
    for mod, data in stats["modalities"].items():
        print(f"--- {mod} ---")
        print(f"samples: {data['sample_count']}")
        print(f"distribution: {data['class_distribution']}\n")

    g = stats["global"]
    print(f"=== global totals ===")
    print(f"total samples: {g['total_samples']}")
    print(f"global distribution: {g['total_class_distribution']}")
    if "labeled_ratio" in g:
        print(f"labeled ratio: {g['labeled_ratio']}")


def summarize_dataset(dataloaders: list, modality_names: list = None, verbose: bool = True) -> dict:
    """Summarizes sample counts and class distributions across multiple dataloaders.

    This function iterates through a list of PyTorch DataLoaders, extracts labels
    from the underlying datasets, and computes both per-modality and global
    class distributions. It also calculates a labeled ratio if unlabeled samples
    (marked as -1) are present.

    Args:
        dataloaders (list[DataLoader]): A list of PyTorch DataLoader objects to summarize.
        modality_names (list[str], optional): Custom names for each modality. 
            Defaults to None, which generates names like "modality_0", "modality_1", etc.
        verbose (bool, optional): If True, prints a formatted summary to the console. 
            Defaults to True.

    Returns:
        dict: A nested dictionary containing:
            - 'modalities': Per-modality sample counts and class distributions.
            - 'global': Aggregated totals across all dataloaders, including 
              'total_samples', 'total_class_distribution', and optionally 'labeled_ratio'.

    Raises:
        AssertionError: If the length of `modality_names` does not match `dataloaders`,
            or if any item in `dataloaders` is not a PyTorch DataLoader.

    Note:
        The function assumes the dataset within the DataLoader returns a tuple 
        where the second element (index 1) is the label.
    """
    if modality_names:
        assert len(modality_names) == len(dataloaders), \
            "The dataloaders and modality names provided should have the same length"
    else:
        modality_names = [f"modality_{i}" for i in range(len(dataloaders))]

    stats = {
        "modalities": {},
        "global": {}
    }

    total_counter = Counter()

    for i, dataloader in enumerate(dataloaders):
        assert isinstance(dataloader, DataLoader), "items must be PyTorch DataLoaders"

        # Extract labels: handles Tensors or raw scalars
        dataset = dataloader.dataset
        labels = [
            data[1].item() if isinstance(data[1], torch.Tensor) else data[1]
            for data in dataset
        ]

        class_counts = dict(sorted(Counter(labels).items()))
        total_samples = sum(class_counts.values())

        # Store modality-specific data
        mod_name = modality_names[i]
        stats["modalities"][mod_name] = {
            "sample_count": total_samples,
            "class_distribution": class_counts
        }

        total_counter.update(class_counts)

    # Global Calculations
    sorted_total = dict(sorted(total_counter.items()))
    total_sum = sum(sorted_total.values())

    stats["global"] = {
        "total_samples": total_sum,
        "total_class_distribution": sorted_total
    }

    # Handle unlabeled data (-1 convention)
    if -1 in sorted_total:
        labeled_count = sum(v for k, v in sorted_total.items() if k != -1)
        stats["global"]["labeled_ratio"] = round(labeled_count / total_sum, 3)

    if verbose:
        _print_summary(stats)

    return stats


def get_samples_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = None,
    device: str = None,
) -> tuple:
    """Read a fixed number of samples from a DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): Source DataLoader.
        num_samples (int, optional): Number of samples to return. If ``None``,
            all samples in the dataset are returned.
        device (str, optional): Device string (e.g., ``'cuda'`` or ``'cpu'``)
            to move tensors to. If ``None``, tensors remain on their original
            device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A ``(samples, labels)`` pair where
        ``samples`` has shape ``(num_samples, ...)`` and ``labels`` has shape
        ``(num_samples,)``.

    Raises:
        Exception: If ``num_samples`` exceeds the total number of samples in
            the dataset.
    """
    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    if num_samples is None:
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
            if len(samples) * batch_size >= num_samples:
                break

    samples = torch.cat(samples, dim=0)[:num_samples]
    labels = torch.cat(labels, dim=0)[:num_samples]

    return samples, labels


def random_split(dataset, length_ratios: list, paired: bool, random_seed: int = None) -> list:
    """Randomly split a dataset or a list of multi-modal datasets.

    For a single dataset the function delegates to
    :func:`torch.utils.data.random_split`. For multiple datasets, splits can
    be performed independently per modality or with paired indices so that
    corresponding samples remain together across modalities.

    Args:
        dataset (torch.utils.data.Dataset or list[torch.utils.data.Dataset]):
            A single dataset or a list of datasets (one per modality).
        length_ratios (list[float]): Split proportions that must sum to
            ``1.0``.
        paired (bool): If ``True`` and ``dataset`` is a list, the same
            index permutation is applied to all modalities so that paired
            samples stay together. Ignored for a single dataset.
        random_seed (int, optional): Seed for reproducibility.
            Defaults to ``None``.

    Returns:
        list: For a single dataset, a list of :class:`~torch.utils.data.Subset`
        objects. For multiple datasets, a list of lists where each inner list
        contains the :class:`~torch.utils.data.Subset` objects for one split.

    Raises:
        ValueError: If the dataset list is empty, if ``paired=True`` and
            datasets differ in length, or if ``length_ratios`` do not sum to
            ``1.0``.
    """
    if isinstance(dataset, Dataset):
        return torch.utils.data.random_split(
            dataset=dataset,
            lengths=length_ratios,
            generator=torch.Generator().manual_seed(random_seed) if random_seed is not None else None,
        )

    if not dataset:
        raise ValueError("Dataset list cannot be empty")

    dataset_length = len(dataset[0])
    if paired and not all(len(ds) == dataset_length for ds in dataset):
        raise ValueError("All datasets must have the same length for paired splitting")

    if not paired:
        splits = list(zip(*[
            torch.utils.data.random_split(
                dataset=ds,
                lengths=length_ratios,
                generator=torch.Generator().manual_seed(random_seed) if random_seed is not None else None,
            )
            for ds in dataset
        ]))
        return splits

    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    indices = torch.randperm(dataset_length, generator=generator).tolist()

    if abs(sum(length_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Length ratios must sum to 1.0, got {sum(length_ratios)}")

    lengths = []
    remaining = dataset_length
    for ratio in length_ratios[:-1]:
        length = int(ratio * dataset_length)
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)

    splits_indices = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        splits_indices.append(indices[start_idx:end_idx])
        start_idx = end_idx

    splits = []
    for split_indices in splits_indices:
        split_datasets = [Subset(ds, split_indices) for ds in dataset]
        splits.append(split_datasets)

    return splits


def _list_of_combinations(n: int) -> list:
    """Generate all non-empty combinations of indices from ``0`` to ``n-1``.

    Args:
        n (int): Total number of items.

    Returns:
        list[tuple[int, ...]]: All non-empty combinations in lexicographic
        order.
    """
    indices = list(range(n))
    return [comb for r in range(1, n + 1) for comb in combinations(indices, r)]


def _find_common_intersection(lists: list) -> list:
    """Find the common intersection of multiple lists.

    Args:
        lists (list[list]): Lists of items to intersect.

    Returns:
        list: Elements present in every sublist. Returns an empty list if
        ``lists`` is empty or the intersection is empty.
    """
    if not lists:
        return []
    return list(set.intersection(*map(set, lists)))


def _unique_intersections(indexes_list: list) -> dict:
    """Compute unique per-combination intersections across modality index lists.

    For each possible non-empty combination of modalities, identifies the sample
    IDs that appear in *all* modalities of that combination but in *none* of the
    remaining modalities.

    Args:
        indexes_list (list[list[str or int]]): Per-modality lists of sample
            identifiers.

    Returns:
        dict[tuple[int, ...], list]: Dictionary mapping each combination tuple
        of modality indices to the sorted list of sample IDs that are unique to
        that combination.
    """
    num_modalities = len(indexes_list)
    combination_dict = {}

    for comb_tuple in _list_of_combinations(num_modalities):
        comb_list = list(comb_tuple)
        current_modalities_indexes = [indexes_list[i] for i in comb_list]
        intersection = set(_find_common_intersection(current_modalities_indexes))

        other_modalities_indices = set(range(num_modalities)) - set(comb_list)
        other_indexes = set()
        for i in other_modalities_indices:
            other_indexes.update(indexes_list[i])

        combination_dict[comb_tuple] = sorted(list(intersection - other_indexes))

    return combination_dict


def multimodal_dataset_split(
    indexes_list: list,
    test_ratio: float = 0.2,
    stratify_by: list = None,
    paired_only: bool = False,
    random_seed: int = None,
) -> tuple:
    """Split a multi-modal dataset into training and test subsets.

    Ensures that samples shared across modalities are handled consistently.
    The split can be stratified by class label to preserve class distributions.

    Args:
        indexes_list (list[list[str or int]]): Per-modality lists of sample
            identifiers.
        test_ratio (float): Proportion of samples to allocate to the test set.
            Defaults to ``0.2``.
        stratify_by (list[list[int]], optional): Per-modality class label lists
            used to stratify the split. Each inner list must have the same
            length as the corresponding entry in ``indexes_list``.
            Defaults to ``None``.
        paired_only (bool): If ``True``, only samples present in *all*
            modalities are considered for the split. Defaults to ``False``.
        random_seed (int, optional): Seed for reproducibility.
            Defaults to ``None``.

    Returns:
        tuple[dict[int, list], dict[int, list]]: A
        ``(train_samples_per_modality, test_samples_per_modality)`` pair.
        Each dictionary maps a modality index to the sorted list of sample IDs
        assigned to that split.

    Raises:
        ValueError: If ``test_ratio`` is outside ``[0, 1]``, if
            ``stratify_by`` has incorrect lengths, or if there are too few
            samples for the requested test ratio.
    """
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1")

    if random_seed is not None:
        random.seed(random_seed)

    num_modalities = len(indexes_list)
    train_samples_per_modality = {i: [] for i in range(num_modalities)}
    test_samples_per_modality = {i: [] for i in range(num_modalities)}

    intersections_dict = _unique_intersections(indexes_list)

    if stratify_by is not None:
        if len(stratify_by) != num_modalities:
            raise ValueError("stratify_by must have the same length as indexes_list")
        for i in range(len(stratify_by)):
            if len(stratify_by[i]) != len(indexes_list[i]):
                raise ValueError(
                    f"stratify_by[{i}] must have the same length as indexes_list[{i}]"
                )
        sample_id_class_dict = {
            k: v
            for row_k, row_v in zip(indexes_list, stratify_by)
            for k, v in zip(row_k, row_v)
        }

    for comb_tuple, unique_samples in intersections_dict.items():
        if paired_only and len(comb_tuple) != num_modalities:
            continue

        if not unique_samples:
            continue

        n_total = len(unique_samples)
        n_test = round(n_total * test_ratio)

        if n_test <= 0 and n_total > 0:
            raise ValueError(
                f"Not enough unique samples ({n_total}) for combination {comb_tuple} "
                f"to create a test set with ratio {test_ratio}."
            )

        if stratify_by is None:
            test_samples = random.sample(unique_samples, n_test)
            train_samples = list(set(unique_samples) - set(test_samples))
        else:
            unique_samples_class = [sample_id_class_dict[i] for i in unique_samples]
            classes_dict = {c: [] for c in set(unique_samples_class)}
            for sidx, s in enumerate(unique_samples):
                classes_dict[unique_samples_class[sidx]].append(s)

            class_nsamples_dict = {
                k: round(n_test * len(v) / n_total)
                for k, v in classes_dict.items()
            }
            test_samples = []
            for ic, n_samples in class_nsamples_dict.items():
                test_samples.extend(random.sample(classes_dict[ic], n_samples))
            test_samples = sorted(test_samples)
            train_samples = list(set(unique_samples) - set(test_samples))

        for i_mod in comb_tuple:
            train_samples_per_modality[i_mod].extend(train_samples)
            test_samples_per_modality[i_mod].extend(test_samples)

    for k in train_samples_per_modality:
        train_samples_per_modality[k] = sorted(train_samples_per_modality[k])
    for k in test_samples_per_modality:
        test_samples_per_modality[k] = sorted(test_samples_per_modality[k])

    return train_samples_per_modality, test_samples_per_modality


def stratified_k_fold(k: int, datasets: list, sample_ids: list = None, random_seed: int = None):
    """Generator for stratified k-fold cross-validation on multi-modal datasets.

    Yields train/test :class:`~torch.utils.data.Subset` pairs for each fold,
    preserving class distribution and (optionally) sample pairing across
    modalities.

    Args:
        k (int): Number of folds. Must be ``>= 2``.
        datasets (list[torch.utils.data.Dataset]): One dataset per modality.
        sample_ids (list[list[str or int]], optional): Per-modality lists of
            sample identifiers used to keep paired samples together. If
            ``None``, samples are treated as unpaired and identified by a
            global sequential index.
        random_seed (int, optional): Seed for reproducibility.
            Defaults to ``None``.

    Yields:
        tuple[list[Subset], list[Subset]]: A ``(train_datasets,
        test_datasets)`` pair for each fold, where each element is a list of
        :class:`~torch.utils.data.Subset` objects (one per modality).

    Raises:
        AssertionError: If ``k < 2``.
    """
    assert k >= 2, "K-fold cross-validation requires k >= 2"

    if sample_ids is None:
        sample_ids = [
            [
                sample_index + sum(len(datasets[j]) for j in range(mi))
                for sample_index, _ in enumerate(ds)
            ]
            for mi, ds in enumerate(datasets)
        ]
    else:
        sample_ids = sample_ids.copy()

    classes = [[sample[1] for sample in ds] for ds in datasets]
    idtoidx = [{sid: i for i, sid in enumerate(modality_ids)} for modality_ids in sample_ids]

    folds = []
    for n in range(k, 0, -1):
        _, test_indexes = multimodal_dataset_split(
            sample_ids,
            test_ratio=1 / n,
            stratify_by=classes,
            paired_only=False,
            random_seed=random_seed,
        )
        folds.append(test_indexes)

        for modality_index in range(len(sample_ids)):
            keep = [
                i for i, idx in enumerate(sample_ids[modality_index])
                if idx not in test_indexes[modality_index]
            ]
            sample_ids[modality_index] = [sample_ids[modality_index][i] for i in keep]
            classes[modality_index] = [classes[modality_index][i] for i in keep]

    for test_index in range(k):
        test_fold_ids = folds[test_index]
        train_folds = folds[:test_index] + folds[test_index + 1:]

        train_fold = defaultdict(list)
        for fold in train_folds:
            for key, value in fold.items():
                train_fold[key].extend(value)
        train_fold_ids = dict(train_fold)

        test_fold_indexes = {
            modality_id: [idtoidx[modality_id][sid] for sid in modality]
            for modality_id, modality in test_fold_ids.items()
        }
        train_fold_indexes = {
            modality_id: [idtoidx[modality_id][sid] for sid in modality]
            for modality_id, modality in train_fold_ids.items()
        }

        test_ds = [Subset(ds, test_fold_indexes[di]) for di, ds in enumerate(datasets)]
        train_ds = [Subset(ds, train_fold_indexes[di]) for di, ds in enumerate(datasets)]

        yield train_ds, test_ds
