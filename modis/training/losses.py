"""
Loss functions for training MODIS.

This module provides clustering and regularization loss functions used during
training, including a Deep Divergence-Based Clustering (DDC) loss and an
entropy regularization loss.

The key components are:

- :class:`DDCLoss`: Implements the DDC loss, which encourages compact and
  separable clusters in the latent space.
- :class:`EntropyLoss`: A regularization loss that prevents the model from
  collapsing all samples into a single cluster.
- :class:`ClusteringLoss`: A combined loss summing the DDC and entropy
  regularization terms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDCLoss(nn.Module):
    """Deep Divergence-Based Clustering (DDC) loss.

    Encourages both compactness and separability of clusters in the latent
    space. Inspired by *"Reconsidering Representation Alignment for
    Multi-View Clustering"* (Trosten et al., 2021).

    Attributes:
        epsilon (float): Small constant added for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-9

    @staticmethod
    def kernel_matrix(x: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """Compute a Gaussian kernel matrix.

        Args:
            x (torch.Tensor): Hidden-layer logits of shape
                ``(n_samples, n_features)``.
            epsilon (float): Small constant for numerical stability.
                Defaults to ``1e-9``.

        Returns:
            torch.Tensor: Kernel matrix of shape ``(n_samples, n_samples)``.
        """
        distances = F.relu(torch.cdist(x, x, p=2) ** 2)
        sigma_squared = 0.15 * torch.median(distances)
        sigma_squared = torch.clamp(sigma_squared, min=epsilon).detach()
        K = torch.exp(-distances / (2 * sigma_squared))
        return K

    def d_cs(self, A: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute the *d* component of the Cauchy-Schwarz (CS) divergence.

        Args:
            A (torch.Tensor): Cluster assignment matrix of shape
                ``(n_samples, n_clusters)``.
            K (torch.Tensor): Similarity (kernel) matrix of shape
                ``(n_samples, n_samples)``.

        Returns:
            torch.Tensor: Scalar *d* value of the CS divergence.
        """
        n_clusters = A.size(1)
        numerator = A.T @ K @ A
        numerator_diag = torch.diag(numerator)
        denom_squared = torch.clamp(
            torch.outer(numerator_diag, numerator_diag), min=self.epsilon
        )
        denominator = torch.sqrt(denom_squared)
        d = (
            2 / (n_clusters * (n_clusters - 1))
            * torch.triu(numerator / denominator, diagonal=1).sum()
        )
        return d

    def calculate_m(self, A: torch.Tensor) -> torch.Tensor:
        """Compute the *m* matrix for the DDC loss.

        The *m* matrix pushes cluster assignment vectors close to the standard
        simplex, encouraging more balanced cluster assignments.

        Args:
            A (torch.Tensor): Cluster assignment logits of shape
                ``(n_samples, n_clusters)``.

        Returns:
            torch.Tensor: The *m* matrix of shape ``(n_samples, n_clusters)``.
        """
        n_clusters = A.size(1)
        e = torch.eye(n_clusters, device=A.device)
        m = torch.exp(-torch.cdist(A, e, p=2) ** 2)
        return m

    def forward(self, hidden_layer: torch.Tensor, aux_layer: torch.Tensor) -> torch.Tensor:
        """Compute the DDC loss.

        Args:
            hidden_layer (torch.Tensor): Logits from the penultimate hidden
                layer, shape ``(n_samples, n_features)``.
            aux_layer (torch.Tensor): Cluster assignment logits from the output
                layer, shape ``(n_samples, n_clusters)``.

        Returns:
            torch.Tensor: Scalar DDC loss value.

        Raises:
            ValueError: If the number of samples in ``hidden_layer`` and
                ``aux_layer`` do not match.
        """
        n_samples, n_clusters = aux_layer.shape
        if hidden_layer.shape[0] != n_samples:
            raise ValueError(
                "Number of samples in hidden_layer and aux_layer must match"
            )

        aux_layer = F.softmax(aux_layer, dim=1)
        K_hid = self.__class__.kernel_matrix(hidden_layer, epsilon=self.epsilon)
        m = self.calculate_m(aux_layer)

        d_hid_alpha = self.d_cs(aux_layer, K_hid)
        d_hid_m = self.d_cs(m, K_hid)

        loss = d_hid_alpha + d_hid_m
        return loss


class EntropyLoss(nn.Module):
    """Entropy regularization loss.

    Prevents degenerate solutions where all samples are assigned to a single
    cluster by encouraging a more uniform distribution of samples across
    clusters.

    Attributes:
        eps (float): Small constant for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute the entropy regularization loss.

        Args:
            logits (torch.Tensor): Output logits of shape
                ``(n_samples, n_clusters)``.

        Returns:
            torch.Tensor: Scalar entropy loss value.
        """
        out_prob = F.softmax(logits, dim=1)
        prob_mean = out_prob.mean(dim=0)
        prob_mean = torch.clamp(prob_mean, min=self.eps)
        loss = torch.sum(prob_mean * torch.log(prob_mean))
        return loss


class ClusteringLoss(nn.Module):
    """Combined clustering loss (DDC + entropy regularization).

    Balances the goals of creating compact and separable clusters
    (:class:`DDCLoss`) with a uniform distribution of samples across clusters
    (:class:`EntropyLoss`).

    Attributes:
        ddc_loss (DDCLoss): Instance of the DDC loss module.
        entropy_loss (EntropyLoss): Instance of the entropy regularization module.
    """

    def __init__(self):
        super().__init__()
        self.ddc_loss = DDCLoss()
        self.entropy_loss = EntropyLoss()

    def forward(self, aux_layer: torch.Tensor, hidden_layer: torch.Tensor) -> torch.Tensor:
        """Compute the combined clustering loss.

        Args:
            aux_layer (torch.Tensor): Cluster assignment logits from the output
                layer, shape ``(n_samples, n_clusters)``.
            hidden_layer (torch.Tensor): Logits from the penultimate hidden
                layer, shape ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Scalar combined DDC and entropy loss.
        """
        loss = self.ddc_loss(hidden_layer, aux_layer) + self.entropy_loss(aux_layer)
        return loss
