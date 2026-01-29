from types import SimpleNamespace

import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
import contextlib


import sys
import ot
from lapsum import soft_permutation_batch

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)



def batched_permutation_matrix(x):
    """
    Generate batched permutation matrices from sorting indices.
    
    Args:
        x: torch.Tensor of shape (B, N) containing values to sort
    
    Returns:
        torch.Tensor of shape (B, N, N) where each (N, N) slice is a permutation matrix
        that represents the sorting permutation for that batch element
    """
    B, N = x.shape
    
    # Get sorting indices
    sort_indices = torch.argsort(x, dim=1)
    
    # Create permutation matrices
    perm_matrices = torch.zeros(B, N, N, dtype=x.dtype).to(x.device)
    
    # Set the appropriate entries to 1
    batch_idx = torch.arange(B).unsqueeze(1).expand(B, N)
    row_idx = torch.arange(N).unsqueeze(0).expand(B, N)
    
    perm_matrices[batch_idx, row_idx, sort_indices] = 1
    
    return perm_matrices


class SWE_Pooling(nn.Module):
    def __init__(self, d_in, num_slices, num_ref_points, alpha_lapsum=10, dual_lr=0.1, eps=10, tau_aggregation=1.0, parallelized = True):
        '''
        Produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size based on sliced-Wasserstein embedding.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_slices: Number of slices
        '''
        super(SWE_Pooling, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_slices = num_slices
        self.dual_lr = dual_lr
        self.eps = eps
        self.tau_aggregation = tau_aggregation
        self.parallelized = parallelized

        uniform_ref = torch.randn(num_ref_points, d_in) # initalize the references using a normal distribution
        self.reference = nn.Parameter(uniform_ref)

        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_slices, bias=False), dim=0)
            
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False


        nn.init.orthogonal_(self.theta.weight_v) # initalize the slicers with a semi-orthogonal matrix

        self.register_buffer('alpha_lapsum', torch.tensor(alpha_lapsum))
        self.register_buffer('lambdas', torch.zeros(num_slices))

        # weights to aggregate the output embedding across ref points
        self.weight = nn.Linear(num_ref_points, 1, bias=False)


        self.ALL_INTERPS = {}

    def forward(self, X, mask=None):
        '''
        Calculates SWE pooling of X over its second dimension (i.e., sequence length)
        
        Input:
            X:  B x N x d_in tensor, containing a batch of B sequences, each containing N embeddings in a d_in-dimensional space
            mask [optional]: B x N binary tensor, with 1 iff the sequence element is valid; used for the case where sequence lengths are different
        Output:
            embeddings: B x (M * num_slices) tensor, containing a batch of B pooled embeddings.
        '''

        B, N, _ = X.shape       
        if mask is None:
            mask = torch.ones(B, N, device=X.device, dtype=torch.bool)
        Xslices = self.get_slice(X)
        alpha = self.alpha_lapsum.to(X.device)

        M, _ = self.reference.shape

        Xslices_sorted, Xind = torch.sort(Xslices, dim=1)

        if M == N:
            Xslices_sorted_interpolated = Xslices_sorted
        else:
            x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
            Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2)
        

        Rslices = self.get_slice(self.reference).expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1) # B x num_slices x M

        # weighting based on lambdas
        lambda_weights = torch.softmax(-self.lambdas/self.tau_aggregation, dim=0).unsqueeze(0).unsqueeze(-1).repeat(B, 1, M) # 1 x num_slices x 1 --> B x num_slices x M
        weighted_embeddings_slicedim = torch.sum(lambda_weights * embeddings, dim=1) # B x M

        # weighting based on a learnable weight vector on the reference elements
        weighted_embeddings_refdim = self.weight(embeddings).sum(-1) # summed over reference elements, output is B x num_slices

        weighted_embeddings = torch.cat([weighted_embeddings_slicedim, weighted_embeddings_refdim], dim=1) # B x (M+num_slices)

        if not self.training:
            return weighted_embeddings, None, None
        
        cost = torch.cdist(X, self.reference.unsqueeze(0).repeat(B, 1, 1), p=2) ** 2 # B x N x M

        ss_r = soft_permutation_batch(Rslices[0].T, alpha=alpha) # L x M --> L x M x M
        ss_r_HARD = batched_permutation_matrix(Rslices[0].T) # L x M --> L x M x M
        
        interp = torch.eye(N, device=X.device, dtype=X.dtype) / N  # N x N

        if not self.parallelized:
            per_slice_distances = []
            per_slice_distances_HARD = []
            

            for b in range(B):
                ss_x = soft_permutation_batch(Xslices[b].T, alpha=alpha)          # (L, N, N)

                ss_x_HARD = batched_permutation_matrix(Xslices[b].T)              # (L, N, N)

                plans   = ss_x.transpose(-1, -2) @ interp @ ss_r
                plans_H = ss_x_HARD.transpose(-1, -2) @ interp @ ss_r_HARD

                cb = cost[b].unsqueeze(0)                                         # (1, N, N) broadcasts over L
                costs_b = torch.sqrt(torch.sum(cb * plans, dim=(-1, -2)))          # (L,)
                costs_b_H = torch.sqrt(torch.sum(cb * plans_H, dim=(-1, -2)))      # (L,)

                per_slice_distances.append(costs_b)
                per_slice_distances_HARD.append(costs_b_H)

            per_slice_distances = torch.stack(per_slice_distances, dim=0).mean(dim=0)          # (L,)
            per_slice_distances_HARD = torch.stack(per_slice_distances_HARD, dim=0).mean(dim=0) # (L,)
        
        else:
            B, N, L = Xslices.shape
            assert M == N

            # Cheaper + correct: torch.cdist supports (B,N,d) vs (M,d) directly
            cost = torch.cdist(X, self.reference, p=2) ** 2   # (B, N, N)

            # (B, L, N)
            Xs = Xslices.transpose(1, 2).contiguous()

            # (L, N) derived from learnable reference + learnable theta
            Rs_ref_LN = self.get_slice(self.reference).T.contiguous()

            invN = 1.0 / N

            per_slice_distances = []
            per_slice_distances_HARD = []

            for l in range(L):
                x_slice = Xs[:, l, :].contiguous()            # (B, N)
                r_row   = Rs_ref_LN[l:l+1, :].contiguous()    # (1, N)

                alpha_val = self.alpha_lapsum.to(device=x_slice.device, dtype=x_slice.dtype)  # Tensor, no grad
                    
                ss_x   = soft_permutation_batch(x_slice, alpha=alpha_val)  # (B, N, N)
                ss_r_1 = soft_permutation_batch(r_row,   alpha=alpha_val)  # (1, N, N)
                ss_r   = ss_r_1.repeat(B, 1, 1)                             # (B, N, N)

                plan = (ss_x.transpose(-1, -2) @ ss_r) * invN                   # (B, N, N)

                dist_l = torch.sqrt((cost * plan).sum(dim=(-1, -2)).clamp_min(0.0)).mean()
                per_slice_distances.append(dist_l)

                # HARD: match order statistics
                idx_x = torch.argsort(x_slice, dim=1)                           # (B, N)
                idx_r = torch.argsort(r_row,   dim=1).repeat(B, 1)              # (B, N)  (no expand)

                b_ids = torch.arange(B, device=X.device).unsqueeze(1).expand(B, N)
                hard_sum = cost[b_ids, idx_x, idx_r].sum(dim=1) * invN
                dist_l_H = torch.sqrt(hard_sum.clamp_min(0.0)).mean()
                per_slice_distances_HARD.append(dist_l_H)

            per_slice_distances = torch.stack(per_slice_distances, dim=0)            # (L,)
            per_slice_distances_HARD = torch.stack(per_slice_distances_HARD, dim=0)  # (L,)



        lambdas_prev = self.lambdas.clone().detach()

        self.lambdas += self.dual_lr * (per_slice_distances_HARD.detach() - self.eps) # hard sorting for lambda updates
        
        self.lambdas.data.clamp_(min=0)

        return weighted_embeddings, per_slice_distances, lambdas_prev

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)
        