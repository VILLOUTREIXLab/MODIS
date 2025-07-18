import torch
import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, normalized_mutual_info_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

def adjust_time(seconds: int) -> str:
    """
    Converts a given number of seconds into a more appropriate time unit

    Args:
        seconds (int): The number of seconds to be converted.

    Return:
        (str): A string representing the time duration in a more convenient unit.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    elif seconds < 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def accuracy(logits, target):
    pred = logits.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).sum().item()
    return correct / logits.size(0)

def calc_classification_metrics(true_labels, pred_labels): # : np.ndarray?
    cm = confusion_matrix(true_labels, pred_labels)
    acc = np.trace(cm) / np.sum(cm)

    balanced_acc = balanced_accuracy_score(true_labels, pred_labels)  # average of recall obtained on each class
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ji = jaccard_score(true_labels, pred_labels, average='macro')    
    ari = adjusted_rand_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Inverse frequency weighting accuracy
    class_weights = dict()
    for class_label in np.unique(true_labels):
        class_weights[class_label] = 1 / np.sum(true_labels == class_label)
    sample_weights = np.array([class_weights[y] for y in true_labels])
    weighted_accuracy = accuracy_score(true_labels, pred_labels, sample_weight=sample_weights)

    return {
        'acc': acc.item(),
        'inv-freq-w-acc': weighted_accuracy,  ### to remove
        'bacc': balanced_acc,
        'nmi': nmi if type(nmi) == float else nmi.item(),
        'ji': ji if type(ji) == float else ji.item(),
        'ari': ari if type(ari) == float else ari.item(),
        'f1': f1 if type(f1) == float else f1.item()
    }

def evaluate_model(model, dataloaders) -> dict:
    """Validate the model on labeled samples"""
    device = model.device
    mse_loss = torch.nn.MSELoss()

    model.eval()
    pred_y = []
    true_y = []
    recon_loss = []
    with torch.no_grad():
        for idx, dl in enumerate(dataloaders):
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
                recon_loss.append(mse_loss(reconstruction, x).cpu())

    true_y = torch.cat(true_y, dim=0).tolist()
    pred_y = torch.cat(pred_y, dim=0).tolist()

    metrics = calc_classification_metrics(true_labels=true_y, pred_labels=pred_y)
    
    recon_loss = torch.stack(recon_loss).mean().item()
    metrics['mse'] = recon_loss

    return metrics

def all_pairs_mse(X: np.ndarray, X_hat: np.ndarray):
    """
    Calculates the mean squared error (MSE) for all pairs of samples between X and X_hat

    Args:
        X (np.ndarray): First input array of shape [n_samples_X, n_features]
        X_hat (np.ndarray): Second input array of shape [n_samples_X_hat, n_features]

    Returns:
        float: The overall mean of the MSE matrix
    """
    # Broadcast to [n_samples_X, n_samples_X_hat, n_features]
    differences = X[:, np.newaxis, :] - X_hat[np.newaxis, :, :]
    
    # Square the differences and average over the feature axis (dim=2)
    mse_matrix = np.mean(differences ** 2, axis=2)
    
    # Return the overall mean of the MSE matrix
    return np.mean(mse_matrix).item()

# Pytorch version
# def all_pairs_mse(X, X_hat):
#     differences = X[:, None, :] - X_hat[None, :, :]  # Broadcast to [n_samples_X, n_samples_X_hat, n_features]
#     mse_matrix = torch.mean(differences ** 2, dim=2)  # Averaging over the feature axis (dim=2)
#     return mse_matrix.mean().item()
