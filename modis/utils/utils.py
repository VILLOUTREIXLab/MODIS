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

## This is counting both directions of the pais, not unique pairs, fix
## includes self-comparisons?
# def all_pairs_mse(X: np.ndarray, X_hat: np.ndarray):
#     """
#     Calculates the mean squared error (MSE) for all pairs of samples between X and X_hat

#     Args:
#         X (np.ndarray): First input array of shape [n_samples_X, n_features]
#         X_hat (np.ndarray): Second input array of shape [n_samples_X_hat, n_features]

#     Returns:
#         float: The overall mean of the MSE matrix
#     """
#     # Broadcast to [n_samples_X, n_samples_X_hat, n_features]
#     differences = X[:, np.newaxis, :] - X_hat[np.newaxis, :, :]
    
#     # Square the differences and average over the feature axis (dim=2)
#     mse_matrix = np.mean(differences ** 2, axis=2)
    
#     # Return the overall mean of the MSE matrix
#     return np.mean(mse_matrix).item()

# Pytorch version
# def all_pairs_mse(X, X_hat):
#     differences = X[:, None, :] - X_hat[None, :, :]  # Broadcast to [n_samples_X, n_samples_X_hat, n_features]
#     mse_matrix = torch.mean(differences ** 2, dim=2)  # Averaging over the feature axis (dim=2)
#     return mse_matrix.mean().item()
