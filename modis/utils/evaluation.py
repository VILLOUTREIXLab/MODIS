import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, normalized_mutual_info_score, balanced_accuracy_score #, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

from modis.utils.io import load_config
from modis.utils.data import get_dataloaders

def accuracy(logits, target):
    """Calculates the accuracy of model predictions.

    Args:
        logits (torch.Tensor): The model's raw output predictions.
        target (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    pred = logits.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / logits.size(0)

def calc_classification_metrics(true_labels, pred_labels):
    """Calculates various classification metrics.

    Args:
        true_labels (np.ndarray): The ground truth labels.
        pred_labels (np.ndarray): The predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics:
              'acc' (accuracy), 'bacc' (balanced accuracy), 'nmi'
              (normalized mutual information), 'ji' (Jaccard index),
              'ari' (Adjusted Rand Index), and 'f1' (F1-score).
    """
    cm = confusion_matrix(true_labels, pred_labels)
    acc = np.trace(cm) / np.sum(cm)

    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)  # average of recall obtained on each class
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ji = jaccard_score(true_labels, pred_labels, average='macro')    
    ari = adjusted_rand_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # # Inverse frequency weighting accuracy (inv-freq-w-acc)
    # class_weights = dict()
    # for class_label in np.unique(true_labels):
    #     class_weights[class_label] = 1 / np.sum(true_labels == class_label)
    # sample_weights = np.array([class_weights[y] for y in true_labels])
    # weighted_accuracy = accuracy_score(true_labels, pred_labels, sample_weight=sample_weights)

    return {
        'acc': acc.item(),
        'bacc': balanced_acc,
        'nmi': nmi if type(nmi) == float else nmi.item(),
        'ji': ji if type(ji) == float else ji.item(),
        'ari': ari if type(ari) == float else ari.item(),
        'f1': f1 if type(f1) == float else f1.item()
    }

def evaluate_model(model, dataloaders) -> dict:
    """Validates the model on labeled samples.

    This function iterates through the provided dataloaders, computes predictions
    and reconstruction losses for labeled samples, and aggregates the results
    to calculate various classification and reconstruction metrics.

    Args:
        model: The trained model to evaluate.
        dataloaders (list): A list of PyTorch DataLoaders, one for each
            modality, containing the data to be evaluated.

    Returns:
        dict: A dictionary containing the evaluation metrics. An empty
              dictionary is returned if no labeled samples are found.
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

                # Reconstruction
                latents = model.get_latents(x, input_modality=idx)
                reconstruction = model.variational_autoencoders[idx].decode(latents)
                recon_loss[idx].append(mse_loss(reconstruction, x).cpu())

    if not true_y:
        return dict()

    true_y = torch.cat(true_y, dim=0).tolist()
    pred_y = torch.cat(pred_y, dim=0).tolist()
    metrics = calc_classification_metrics(true_labels=true_y, pred_labels=pred_y)

    modal_mse = []
    for i in range(len(recon_loss)):
        modal_mse.append(torch.stack(recon_loss[i]).mean().item())
    recon_loss = torch.stack([batch_mse for i in recon_loss for batch_mse in recon_loss[i]]).mean().item()
    
    metrics['modal_mse'] = modal_mse
    metrics['mse'] = recon_loss

    return metrics

def evaluate_checkpoint(
    config,
    checkpoint_dir: Path,
    checkpoint_opt: str,
    train_dataloaders,
    val_dataloaders,
    metrics_data: dict
) -> None:
    """Evaluates a model loaded from a specified checkpoint.

    This function loads a model from a checkpoint file and evaluates its
    performance on both the training and validation datasets. The results are
    printed to the console and stored in a metrics dictionary.

    Args:
        config (DictConfig): The model's configuration.
        checkpoint_dir (Path): The directory containing the checkpoint file.
        checkpoint_opt (str): The name of the checkpoint ('best' or 'latest').
        train_dataloaders (list): A list of dataloaders for the training data.
        val_dataloaders (list): A list of dataloaders for the validation data.
        metrics_data (dict): A dictionary to store the evaluation results.
    """
    from modis import Model

    model = Model(config)
    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_opt}.pth"
    model.load_from_checkpoint(checkpoint_file, verbose=False)

    print(f"==> Evaluation metrics on train dataset for {checkpoint_opt} checkpoint")
    metrics = evaluate_model(model, train_dataloaders)
    metrics_data[checkpoint_opt]['train'] = metrics
    for metric_name, metric_value in metrics.items():
        if type(metric_value) == list:
            print(f"{metric_name}: {[round(v,4) for v in metric_value]}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")

    if val_dataloaders is not None:
        print(f"==> Evaluation metrics on validation dataset for {checkpoint_opt} checkpoint")
        metrics = evaluate_model(model, val_dataloaders)
        metrics_data[checkpoint_opt]['validation'] = metrics
        for metric_name, metric_value in metrics.items():
            if type(metric_value) == list:
                print(f"{metric_name}: {[round(v,4) for v in metric_value]}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")

def launch_checkpoints_evaluation(
    train_datasets,
    val_datasets,
    checkpoint_dir: Path,
) -> None:
    """Launches the evaluation of the latest and best model checkpoints.

    This function loads the model's configuration and then, based on the
    `save_checkpoint_latest` and `save_checkpoint_best` flags, calls
    `evaluate_checkpoint` for each enabled checkpoint. Finally, it saves the
    evaluation metrics to a JSON file.

    Args:
        train_datasets (list): A list of PyTorch Datasets for the training data.
        val_datasets (list): A list of PyTorch Datasets for the validation data.
        checkpoint_dir (Path): The directory where the checkpoints are saved.
    """
    metrics_data = {'latest': dict(), 'best': dict()}

    config = load_config(checkpoint_dir / 'config.yaml')
    if not config.save_checkpoint_latest and not config.save_checkpoint_best:
        return

    train_dataloaders = get_dataloaders(train_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
    val_dataloaders = None
    if val_datasets is not None:
        val_dataloaders = get_dataloaders(val_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)

    if config.save_checkpoint_best:
        evaluate_checkpoint(config, checkpoint_dir, 'best', train_dataloaders, val_dataloaders, metrics_data)
        print()

    if config.save_checkpoint_latest:
        evaluate_checkpoint(config, checkpoint_dir, 'latest', train_dataloaders, val_dataloaders, metrics_data)

    try:
        with open(checkpoint_dir / f"checkpoints_evaluation_metrics.json", 'w', encoding='utf-8') as json_file:
            json.dump(metrics_data, json_file, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving evaluation metrics file: {e}")

def avg_mse_all_pairs(X, X_hat):
    """Computes the average MSE between all pairs of vectors from X and X_hat.

    This function uses broadcasting to efficiently calculate the mean squared
    error between every vector in `X` and every vector in `X_hat`.

    Args:
        X (torch.Tensor): The first tensor of vectors.
        X_hat (torch.Tensor): The second tensor of vectors.

    Returns:
        float: The average MSE across all pairs.
    """
    differences = X[:, None, :] - X_hat[None, :, :]  # Broadcast to [n_samples_X, n_samples_X_hat, n_features]
    mse_matrix = torch.mean(differences ** 2, dim=2)  # Averaging over the feature axis (dim=2)
    return mse_matrix.mean().item()