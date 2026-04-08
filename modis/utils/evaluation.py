"""
Evaluation utilities for MODIS.

This module provides functions for computing classification metrics,
evaluating trained models on datasets, and running batch checkpoint
evaluations.
"""
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    jaccard_score,
    f1_score,
    normalized_mutual_info_score,
    balanced_accuracy_score,
)
from sklearn.metrics.cluster import adjusted_rand_score

from modis.utils.io import load_config
from modis.utils.data import get_dataloaders


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy from logits and ground-truth labels.

    Args:
        logits (torch.Tensor): Raw model output of shape
            ``(batch_size, n_classes)``.
        target (torch.Tensor): Ground-truth label indices of shape
            ``(batch_size,)``.

    Returns:
        float: Fraction of correctly predicted samples in ``[0, 1]``.
    """
    pred = logits.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / logits.size(0)


def calc_classification_metrics(true_labels, pred_labels) -> dict:
    """Compute a suite of classification metrics.

    Args:
        true_labels (array-like): Ground-truth label array of shape
            ``(n_samples,)``.
        pred_labels (array-like): Predicted label array of shape
            ``(n_samples,)``.

    Returns:
        dict: Dictionary with the following keys:

        - ``'acc'`` (float): Overall accuracy.
        - ``'bacc'`` (float): Balanced accuracy (macro-averaged recall).
        - ``'nmi'`` (float): Normalized Mutual Information.
        - ``'ji'`` (float): Macro-averaged Jaccard Index.
        - ``'ari'`` (float): Adjusted Rand Index.
        - ``'f1'`` (float): Weighted-average F1-score.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    acc = np.trace(cm) / np.sum(cm)

    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ji = jaccard_score(true_labels, pred_labels, average='macro')
    ari = adjusted_rand_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    return {
        'acc': acc.item(),
        'bacc': balanced_acc,
        'nmi': nmi if type(nmi) == float else nmi.item(),
        'ji': ji if type(ji) == float else ji.item(),
        'ari': ari if type(ari) == float else ari.item(),
        'f1': f1 if type(f1) == float else f1.item(),
    }


def evaluate_model(model, dataloaders) -> dict:
    """Evaluate a trained model on labeled samples from multiple dataloaders.

    Iterates through the dataloaders, computes predictions and reconstruction
    losses for all labeled samples, and aggregates results across modalities.
    Samples with label ``-1`` are treated as unlabeled and skipped.

    Args:
        model: The trained :class:`~modis.nn.Model` to evaluate.
        dataloaders (list[torch.utils.data.DataLoader]): One DataLoader per
            modality.

    Returns:
        dict: Evaluation metric dictionary. Returns an empty dict if no
        labeled samples are found. Otherwise contains all keys from
        :func:`calc_classification_metrics` plus:

        - ``'mse'`` (float): Mean reconstruction MSE across all modalities.
        - ``'modal_mse'`` (list[float]): Per-modality mean reconstruction MSE.
    """
    device = model.device
    mse_loss = torch.nn.MSELoss()

    model.eval()
    pred_y = []
    true_y = []
    recon_loss = dict()
    with torch.no_grad():
        for idx, dl in enumerate(dataloaders):
            recon_loss[idx] = []
            for data in dl:
                x, y = data[0].to(device), data[1].to(device)
                label_mask = torch.tensor([True if label != -1 else False for label in y])

                if sum(label_mask) == 0:
                    continue

                x = x[label_mask]
                y = y[label_mask]

                if x.size(0) == 0:
                    print(f"[!] No labeled samples found in modality {idx}, skipping")
                    continue

                pred_y.append(model.predict(x, input_modality=idx))
                true_y.append(y.view(-1))

                latents = model.get_latents(x, input_modality=idx)
                reconstruction = model.variational_autoencoders[idx].decode(latents)
                recon_loss[idx].append(mse_loss(reconstruction, x).cpu())

    if not true_y:
        return dict()

    true_y = torch.cat(true_y, dim=0).tolist()
    pred_y = torch.cat(pred_y, dim=0).tolist()
    metrics = calc_classification_metrics(true_labels=true_y, pred_labels=pred_y)

    modal_mse = [torch.stack(recon_loss[i]).mean().item() for i in range(len(recon_loss))]
    recon_loss_mean = torch.stack(
        [batch_mse for i in recon_loss for batch_mse in recon_loss[i]]
    ).mean().item()

    metrics['modal_mse'] = modal_mse
    metrics['mse'] = recon_loss_mean

    return metrics


def evaluate_checkpoint(
    config,
    checkpoint_dir: Path,
    checkpoint_opt: str,
    train_dataloaders,
    val_dataloaders,
    metrics_data: dict,
) -> None:
    """Evaluate a model loaded from a specific checkpoint.

    Loads the model weights from the given checkpoint, runs evaluation on
    both training and (optionally) validation data, prints results to stdout,
    and stores them in ``metrics_data``.

    Args:
        config (omegaconf.DictConfig): Model configuration.
        checkpoint_dir (pathlib.Path): Directory containing the checkpoint
            file.
        checkpoint_opt (str): Checkpoint variant to load; either ``'best'``
            or ``'latest'``.
        train_dataloaders (list[torch.utils.data.DataLoader]): Training data
            loaders.
        val_dataloaders (list[torch.utils.data.DataLoader] or None):
            Validation data loaders, or ``None`` if no validation set is
            available.
        metrics_data (dict): Dictionary in which evaluation results are
            stored under ``metrics_data[checkpoint_opt]['train']`` and
            ``metrics_data[checkpoint_opt]['validation']``.
    """
    from modis import Model

    model = Model(config)
    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_opt}.pth"
    model.load_from_checkpoint(checkpoint_file, verbose=False)

    print(f"==> Evaluation metrics on train dataset for {checkpoint_opt} checkpoint")
    metrics = evaluate_model(model, train_dataloaders)
    metrics_data[checkpoint_opt]['train'] = metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, list):
            print(f"{metric_name}: {[round(v, 4) for v in metric_value]}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")

    if val_dataloaders is not None:
        print(f"==> Evaluation metrics on validation dataset for {checkpoint_opt} checkpoint")
        metrics = evaluate_model(model, val_dataloaders)
        metrics_data[checkpoint_opt]['validation'] = metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, list):
                print(f"{metric_name}: {[round(v, 4) for v in metric_value]}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")


def launch_checkpoints_evaluation(
    train_datasets,
    val_datasets,
    checkpoint_dir: Path,
) -> None:
    """Evaluate the best and/or latest checkpoints and save results to JSON.

    Loads the run's configuration, then—based on ``save_checkpoint_latest``
    and ``save_checkpoint_best`` flags—calls :func:`evaluate_checkpoint` for
    each enabled checkpoint. Results are written to
    ``checkpoints_evaluation_metrics.json`` inside ``checkpoint_dir``.

    Args:
        train_datasets (list[torch.utils.data.Dataset]): Training datasets,
            one per modality.
        val_datasets (list[torch.utils.data.Dataset] or None): Validation
            datasets, one per modality, or ``None``.
        checkpoint_dir (pathlib.Path): Directory where checkpoint files and
            the configuration YAML are stored.
    """
    metrics_data = {'latest': dict(), 'best': dict()}

    config = load_config(checkpoint_dir / 'config.yaml')
    if not config.save_checkpoint_latest and not config.save_checkpoint_best:
        return

    train_dataloaders = get_dataloaders(
        train_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False
    )
    val_dataloaders = None
    if val_datasets is not None:
        val_dataloaders = get_dataloaders(
            val_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False
        )

    if config.save_checkpoint_best:
        evaluate_checkpoint(config, checkpoint_dir, 'best', train_dataloaders, val_dataloaders, metrics_data)
        print()

    if config.save_checkpoint_latest:
        evaluate_checkpoint(config, checkpoint_dir, 'latest', train_dataloaders, val_dataloaders, metrics_data)

    try:
        with open(checkpoint_dir / "checkpoints_evaluation_metrics.json", 'w', encoding='utf-8') as json_file:
            json.dump(metrics_data, json_file, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving evaluation metrics file: {e}")


def avg_mse_all_pairs(X: torch.Tensor, X_hat: torch.Tensor) -> float:
    """Compute the average MSE between all pairs of vectors from two tensors.

    Uses broadcasting to efficiently calculate the mean squared error between
    every vector in ``X`` and every vector in ``X_hat``.

    Args:
        X (torch.Tensor): First tensor of shape ``(n, d)``.
        X_hat (torch.Tensor): Second tensor of shape ``(m, d)``.

    Returns:
        float: Average MSE across all ``n × m`` pairs.
    """
    differences = X[:, None, :] - X_hat[None, :, :]
    mse_matrix = torch.mean(differences ** 2, dim=2)
    return mse_matrix.mean().item()
