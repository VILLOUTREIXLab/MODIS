"""
This module provides various loss functions for training MODIS,
including a Deep Divergence-Based Clustering (DDC) loss and an entropy
regularization loss.

The key components are:
    - `DDCLoss`: Implements the DDC loss, which is used to encourage
      compact and separable clusters in the latent space.
    - `EntropyLoss`: A regularization loss that prevents the model from
      assigning all samples to a single cluster.
    - `ClusteringLoss`: A combined loss that sums the DDC loss and the
      entropy regularization loss for the main clustering objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDCLoss(nn.Module):
    """
    Calculate the DDC loss (Deep Divergence-Based Approach to Clustering).

    This loss function, inspired by "Reconsidering Representation Alignment
    for Multi-View Clustering" by Trosten et al. (2021), encourages both
    compactness and separability of clusters in the latent space.

    Attributes:
        epsilon (float): A small value added for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-9

    @staticmethod
    def kernel_matrix(x, epsilon=1e-9):
        """
        Compute a Gaussian kernel matrix.

        Args:
            x (torch.Tensor): Hidden layer logits (n_samples x n_features).
            epsilon (float): A small value for numerical stability.
        
        Returns:
            torch.Tensor: The kernel matrix (n_samples x n_samples).
        """
        distances = F.relu(torch.cdist(x, x, p=2)**2)

        sigma_squared = 0.15 * torch.median(distances)
        sigma_squared = torch.clamp(sigma_squared, min=epsilon).detach()

        K = torch.exp(-distances / (2 * sigma_squared))

        return K

    def d_cs(self, A, K):
        """
        Calculate the d component of the Cauchy-Schwarz (CS) divergence.
        
        Args:
            A (torch.Tensor): Cluster assignments or m matrix (n_samples x n_clusters).
            K (torch.Tensor): Similarity matrix (n_samples x n_samples).
        
        Returns:
            torch.Tensor: The d value of the CS divergence.
        """
        n_clusters = A.size(1)

        numerator = A.T @ K @ A

        numerator_diag = torch.diag(numerator)
        denom_squared = torch.clamp(torch.outer(numerator_diag, numerator_diag), min=self.epsilon)
        denominator = torch.sqrt(denom_squared)
        
        d = 2 / (n_clusters * (n_clusters - 1)) * torch.triu(numerator / denominator, diagonal=1).sum()

        return d

    def calculate_m(self, A):
        """
        Calculate the m matrix for the DDC loss.
        
        The m matrix pushes the cluster assignment vectors close to the
        standard simplex, encouraging more even cluster assignments.

        Args:
            A (torch.Tensor): Cluster assignments logits (n_samples x n_clusters).
        
        Returns:
            torch.Tensor: The m matrix (n_samples x n_clusters).
        """
        n_clusters = A.size(1)
        e = torch.eye(n_clusters, device=A.device)
        m = torch.exp(-torch.cdist(A, e, p=2)**2)
        return m

    def forward(self, hidden_layer, aux_layer):
        """
        Compute the DDC loss.

        Args:
            hidden_layer (torch.Tensor): Logits from the hidden layer previous
                                         to output (n_samples x n_features).
            aux_layer (torch.Tensor): Logits of the output layer (cluster
                                      assignments) (n_samples x n_clusters).
        
        Returns:
            torch.Tensor: The DDC loss value.
        """
        n_samples, n_clusters = aux_layer.shape
        if hidden_layer.shape[0] != n_samples:
            raise ValueError("Number of samples in hidden_layer and aux_layer must match")
        
        aux_layer = F.softmax(aux_layer, dim=1)  # Crisp cluster assignments

        K_hid = self.__class__.kernel_matrix(hidden_layer, epsilon=self.epsilon)
        m = self.calculate_m(aux_layer)

        d_hid_alpha = self.d_cs(aux_layer, K_hid)  # It requires clusters to be separable and compact
        #triu_AAT = torch.sum(torch.triu(aux_layer @ aux_layer.T, diagonal=1))  # Encourages cluster assignment vectors to be orthogonal
        d_hid_m = self.d_cs(m, K_hid)  # Pushes the cluster assignment vectors close to the standard simplex

        #loss = d_hid_alpha + triu_AAT + d_hid_m
        loss = d_hid_alpha + d_hid_m  # Tang et al., 2023
        #loss = d_hid_m
        
        return loss

class EntropyLoss(nn.Module):
    """
    Computes the entropy regularization to avoid the assignment of only a subset
    of the total clusters.

    This loss encourages a more uniform distribution of samples across clusters,
    preventing degenerate solutions where a single cluster dominates.

    Attributes:
        eps (float): A small value for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, logits):
        """
        Calculates the entropy loss.

        Args:
            logits (torch.Tensor): Logits from the output layer.

        Returns:
            torch.Tensor: The calculated entropy loss.
        """
        out_prob = F.softmax(logits, dim=1)
        
        prob_mean = out_prob.mean(dim=0)
        prob_mean = torch.clamp(prob_mean, min=self.eps)

        loss = torch.sum(prob_mean * torch.log(prob_mean))

        return loss

class ClusteringLoss(nn.Module):
    """
    Combines the DDC loss and entropy regularization loss.
    
    This class provides a comprehensive loss function for clustering,
    balancing the goals of creating compact, separable clusters with a
    uniform distribution of samples across them.

    Attributes:
        ddc_loss (DDCLoss): An instance of the DDCLoss module.
        entropy_loss (EntropyLoss): An instance of the EntropyLoss module.
    """

    def __init__(self):
        super().__init__()
        self.ddc_loss = DDCLoss()
        self.entropy_loss = EntropyLoss()

    def forward(self, aux_layer, hidden_layer):
        """
        Computes the total clustering loss.

        Args:
            aux_layer (torch.Tensor): Logits from the output layer.
            hidden_layer (torch.Tensor): Logits from the hidden layer.

        Returns:
            torch.Tensor: The combined DDC and entropy loss.
        """
        loss = self.ddc_loss(hidden_layer, aux_layer) + self.entropy_loss(aux_layer)
        return loss
