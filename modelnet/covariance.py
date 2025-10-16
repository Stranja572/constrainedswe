import torch

def covariance_pool(X: torch.Tensor):
    """
    Covariance pooling: computes the mean and the upper triangle of the covariance matrix.

    Args:
        X: torch.Tensor of shape (B, N, d)
           B = batch size, N = number of samples, d = feature dimension
    
    Returns:
        mean: torch.Tensor of shape (B, d)
        cov_upper: torch.Tensor of shape (B, d*(d+1)//2)
                   Upper-triangular elements of covariance matrix for each batch
    """
    B, N, d = X.shape

    # Compute mean (B, d)
    mean = X.mean(dim=1)

    # Center data
    X_centered = X - mean.unsqueeze(1)

    # Covariance (B, d, d)
    cov = torch.matmul(X_centered.transpose(1, 2), X_centered) / (N - 1)

    # Extract upper-triangular elements
    idx = torch.triu_indices(d, d)
    cov_upper = cov[:, idx[0], idx[1]]  # (B, d*(d+1)//2)

    return torch.concatenate((mean,cov_upper),1)