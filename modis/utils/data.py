from collections import Counter

from torch.utils.data import Dataset, DataLoader


def get_dataloaders(
    datasets: list[Dataset],
    batch_size: int,
    drop_last: bool = True,
    shuffle: bool = True
) -> list[DataLoader]:
    """Return a dataloader for each dataset"""
    dataloaders = [DataLoader(ds, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
                   for ds in datasets]
    return dataloaders

def summarize_dataset(dataloaders: list[DataLoader], modality_names: list | None = None) -> None:
    """
    Print the total number of samples in a dataloader and their class distribution.
    
    Note: Requires labeled datasets
    """
    if modality_names:
        assert len(modality_names) == len(dataloaders), "The dataloaders and modality names provided should have the same length"

    total_samples = Counter()
    for i, dataloader in enumerate(dataloaders):
        assert isinstance(dataloader, DataLoader), f"You must provide a list of Pytorch DataLoaders"

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
