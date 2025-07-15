import torch
import torch.nn as nn
import torch.nn.functional as F

class DDCLoss(nn.Module):
    """
    Calculate the DDC loss (Deep Divergence-Based Approach to Clustering)

    Reconsidering Representation Alignment for Multi-View Clustering
    Trosten et al., 2021.
    https://doi.org/10.1109/CVPR46437.2021.00131.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-9

    @staticmethod
    def kernel_matrix(x, epsilon=1e-9):
        """
        Compute a Gaussian kernel matrix

        Args:
            x (torch.Tensor): Hidden layer logits (n_samples x n_features)
        
        Return:
            (torch.Tensor): Kernel matrix (n_samples x n_samples)
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
            A (torch.Tensor): Cluster assignments or m matrix (n_samples x n_clusters)
            K (torch.Tensor): Similarity matrix (n_samples x n_samples)
        
        Return:
            (torch.Tensor): d value
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
        Calculate the m matrix for the DDC loss
        
        Args:
            A (torch.Tensor): Cluster assignments logits (n_samples x n_clusters)
        
        Return:
            (torch.Tensor): m matrix (n_samples x n_clusters)
        """
        n_clusters = A.size(1)
        e = torch.eye(n_clusters, device=A.device)
        m = torch.exp(-torch.cdist(A, e, p=2)**2)
        return m

    def forward(self, hidden_layer, aux_layer):
        """
        Compute the DDC loss

        Args:
            hidden_layer (torch.Tensor): Logits from the hidden layer previous to output (n_samples x n_features)
            aux_layer (torch.Tensor): Logits of output layer (cluster assignments) (n_samples x n_clusters)
        
        Return:
            (torch.Tensor): DDC loss value
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
    Computes the entropy regularization to avoid the assignment of only a subset of the total clusters
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, logits):
        out_prob = F.softmax(logits, dim=1)
        
        prob_mean = out_prob.mean(dim=0)
        prob_mean = torch.clamp(prob_mean, min=self.eps)

        loss = torch.sum(prob_mean * torch.log(prob_mean))

        return loss


def joint_probability(logits1, logits2):
    # Convert logits to probabilities
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)
    
    # Compute outer product
    joint_probs = torch.bmm(probs1.unsqueeze(2), probs2.unsqueeze(1))
    
    # Normalize
    joint_probs = joint_probs / joint_probs.sum(dim=(1,2), keepdim=True)
    
    return joint_probs

def joint_entropy(logits1, logits2):
    # Convert logits to probabilities
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)
    
    # Compute outer product for joint probability
    joint_probs = torch.bmm(probs1.unsqueeze(2), probs2.unsqueeze(1))
    
    # Normalize
    joint_probs = joint_probs / joint_probs.sum(dim=(1,2), keepdim=True)
    
    # Compute entropy
    entropy = torch.sum(joint_probs * torch.log2(joint_probs + 1e-12), dim=(1,2))
    
    return entropy

class ClusteringLoss(nn.Module):
    """
    Computes the entropy regularization to avoid the assignment of only a subset of the total clusters
    """

    def __init__(self):
        super().__init__()
        self.ddc_loss = DDCLoss()
        self.entropy_loss = EntropyLoss()

    def forward(self, adv_layer, aux_layer, hidden_layer):
        loss = self.ddc_loss(hidden_layer, aux_layer) + self.entropy_loss(aux_layer)
        #loss = self.entropy_loss(joint_probability(adv_layer, aux_layer))
        #loss = self.ddc_loss(hidden_layer, aux_layer) + self.entropy_loss(adv_layer) + self.entropy_loss(aux_layer)
        #loss = self.ddc_loss(hidden_layer, aux_layer) + joint_entropy(adv_layer, aux_layer).mean()
        return loss
