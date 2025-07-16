from collections import Counter

import torch

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