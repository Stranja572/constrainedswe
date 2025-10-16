import torch
import torch.nn as nn

import torch

#Wasserstein embedding (with calculating the plan, Monge couplings though)
def sinkhorn_torch(x, y, p=2, w_x=None, w_y=None, eps=0.1, max_iters=100, stop_thresh=1e-5, verbose=False):
    """
    Compute the entropy-regularized optimal transport (Sinkhorn) plan
    between two point clouds using PyTorch only.
    
    Args:
        x: (B?,) n x d or B x n x d tensor of source points
        y: (B?,) m x d or B x m x d tensor of target points
        p: cost exponent (1=Manhattan, 2=Euclidean)
        w_x, w_y: optional weights (default = uniform)
        eps: regularization parameter
        max_iters: maximum Sinkhorn iterations
        stop_thresh: tolerance for convergence
    Returns:
        P: transport plan (n x m) or (B x n x m)
    """
    # Allow for optional batching
    batched = x.dim() == 3
    if not batched:
        x, y = x.unsqueeze(0), y.unsqueeze(0)

    B, n, d = x.shape
    m = y.shape[1]

    # Uniform weights if not provided
    if w_x is None:
        w_x = torch.ones(B, n, device=x.device) / n
    if w_y is None:
        w_y = torch.ones(B, m, device=x.device) / m

    w_x = w_x.view(B, n)
    w_y = w_y.view(B, m)

    # Compute cost matrix (B x n x m)
    C = torch.cdist(x, y, p=p)

    # Initialize dual variables
    u = torch.zeros_like(w_x)
    v = torch.zeros_like(w_y)

    # Sinkhorn iterations
    for _ in range(max_iters):
        u_prev, v_prev = u.clone(), v.clone()

        u = eps * (torch.log(w_x + 1e-8) - torch.logsumexp((-C + v.unsqueeze(1)) / eps, dim=2))
        v = eps * (torch.log(w_y + 1e-8) - torch.logsumexp((-C + u.unsqueeze(2)) / eps, dim=1))

        if (u - u_prev).abs().max() < stop_thresh and (v - v_prev).abs().max() < stop_thresh:
            break

    # Transport plan
    P = torch.exp((-C + u.unsqueeze(2) + v.unsqueeze(1)) / eps)

    return P if batched else P.squeeze(0)



class LOTSinkhorn(nn.Module):
    def __init__(self, d_in,  num_ref_points):
        '''
        The LOTSinkhorm module that produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size.
        Inputs:            
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_projections: Number of slices        
        '''
        super(LOTSinkhorn, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        
        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, d_in) #m x d_in (reference points are in the original space here)
        self.reference = nn.Parameter(uniform_ref) 
        self.sinkhorn_args = {
            'max_iters': 100,
            'stop_thresh': 1e-5,
            'eps': 1e-3,
            'verbose': False,
        }
                        
    def forward(self, X):
        '''
        Calculates Wasserstein Embedding with respect to the reference points.

        Inputs:
            X: B x N x d_in (batch of sets)
        Outputs:
            embeddings: B x N_ref x d_in
        '''

        B, N, dn = X.shape
        N_ref = self.num_ref_points  

        # Expand reference points for the batch
        reference_expanded = self.reference.unsqueeze(0).expand(B, -1, -1)  # B x N_ref x d_in

        # Compute the batched Sinkhorn transport plans: B x N_ref x N
        plan = sinkhorn_torch(reference_expanded, X, **self.sinkhorn_args)

        # Normalize and compute barycenters: B x N_ref x d_in
        denominator = plan.sum(dim=2, keepdim=True) + 1e-8  # B x N_ref x 1
        barycenters = torch.bmm(plan, X) / denominator      # B x N_ref x d_in

        # Shift by the reference points (already broadcasted)
        embeddings = barycenters - self.reference.unsqueeze(0)  # B x N_ref x d_in

        return embeddings
