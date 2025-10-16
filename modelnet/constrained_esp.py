import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import math
import numpy as np           # NEW
import ot                    # pip install POT   NEW
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSort_p2(torch.nn.Module):
    def __init__(self, tau=1.0):#, hard=False):
        super(SoftSort_p2, self).__init__()
        # self.hard = hard
        self.tau = tau

    def forward(self, scores):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted_ = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = ((scores.transpose(1, 2) - sorted_) ** 2).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        # if self.hard:
        #     P = torch.zeros_like(P_hat, device=P_hat.device)
        #     P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
        #     P_hat = (P - P_hat).detach() + P_hat
        return P_hat

class ConstrainedExpectedSlicedPlan(nn.Module):
    def __init__(self, d_in,  num_ref_points, num_projections, tau_softsort=1,temperature=1.0):
        '''
        The ConstrainedExpectedSlicedPlan module that produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size.
        Inputs:            
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_projections: Number of slices        
        '''
        super(ConstrainedExpectedSlicedPlan, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_projections = num_projections
        
        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, d_in) #m x d_in (reference points are in the original space here)
        self.reference = nn.Parameter(uniform_ref) 
        self.temperature = temperature  # Temperature for softmax for Expected Slices 
        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        if False:#num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False
        

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_ref_points))
        nn.init.xavier_uniform_(self.weight)

        self.softsort = SoftSort_p2(tau=tau_softsort)

    def forward(self, X):
        '''
        Calculates GSW between two empirical distributions.
    
        '''

        B, N, dn = X.shape
        N_ref = self.num_ref_points  # number of reference points
        L = self.num_projections  # number of slices
        # Slice X and reference
        Xslices = self.get_slice(X)                          # B x N x L
        Rslices = self.get_slice(self.reference)             # N_ref x L
        Rslices = Rslices.unsqueeze(0).expand(B, -1, -1)     # B x N_ref x L

        # Initialize plan with correct shape: B x N_ref x N x L
        plan = torch.zeros(B, N_ref, N, L, device=X.device)

        if N == N_ref:
            # --- One-to-one mapping case ---
            # Sort X slices along N
            Xslices_sorted, Xind = torch.sort(Xslices, dim=1)    # B x N x L
            _, Rind = torch.sort(Rslices, dim=1)                 # B x N_ref x L (=B x N x L)

            # Create indices
            b_idx = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, N, L)   # B x N x L
            l_idx = torch.arange(L, device=X.device).view(1, 1, L).expand(B, N, L)   # B x N x L

            # Scatter: For each batch b and slice l, map Xind → Rind with uniform weight 1/N
            plan[b_idx, Rind, Xind, l_idx] += 1.0 / N

        else:
            # --- Interpolated transport plan case ---
            x = np.linspace(0,1,N_ref) # --- IGNORE ---
            y = np.linspace(0,1,N) # --- IGNORE ---
            _, P = ot.emd2_1d(x,y,log=True) # --- IGNORE ---
            P_interp = torch.tensor(P['G'], dtype=torch.float32).to(X.device)  # Convert to tensor for use in the model
            # Sort X and reference indices
            _, Xind = torch.sort(Xslices, dim=1)  # B x N x L
            _, Rind = torch.sort(Rslices, dim=1)  # B x N_ref x L

            # Expand P_interp (N_ref x N) → B x N_ref x N x L
            P_expanded = P_interp.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, L)

            # Indices for scatter
            b_idx = torch.arange(B, device=X.device).view(B, 1, 1, 1).expand(B, N_ref, N, L)
            l_idx = torch.arange(L, device=X.device).view(1, 1, 1, L).expand(B, N_ref, N, L)

            # Expand Rind and Xind to match shape: B x N_ref x N x L
            Rind_exp = Rind.unsqueeze(2).expand(-1, -1, N, -1)         # B x N_ref x N x L
            Xind_exp = Xind.unsqueeze(1).expand(-1, N_ref, -1, -1)     # B x N_ref x N x L

            # Scatter P_interp into plan
            plan.index_put_((b_idx, Rind_exp, Xind_exp, l_idx), P_expanded, accumulate=True)

        # --- Compute per-slice distances ---
        ref_expanded = self.reference.unsqueeze(0).expand(B, -1, -1)   # B x N_ref x D
        cost = torch.cdist(ref_expanded, X, p=2)                       # B x N_ref x N

        # exact_dist: total distance per slice
        exact_dist = (cost.unsqueeze(3) * plan).sum(dim=(1, 2))         # B x L

        # Weights across slices
        weights = torch.softmax(-exact_dist / self.temperature, dim=1)  # B x L

        # Weighted plan (across slices)
        expected_plan = (plan * weights.view(B, 1, 1, L)).sum(dim=3)    # B x N_ref x N

        # --- Compute barycenter ---
        # Denominator: total transport weight for each reference point
        denominator = expected_plan.sum(dim=2, keepdim=True) + 1e-8     # B x N_ref x 1

        # Weighted barycenter (reference transported into X space)
        barycenter = torch.bmm(expected_plan, X) / denominator           # B x N_ref x D
        embeddings = barycenter- self.reference.unsqueeze(0)  # B x N_ref x D, centered around reference

        if self.training:
            #For one slice, our dist_l is the mean of the SWGG cost going from reference slice distribution to the N token distribution (of one sample) for all samples
            #Basically we are minimizing the average SWGG per slice (across a mini batch), not looking at it per sample
            per_slice_distances = []
            for l in range(self.num_projections):
                

                x_slice = Xslices[:, :, l] # B×N
                r_slice = Rslices[:, :, l] # B×N_ref
                #print(f"[slice {l}] x_slice: {x_slice.shape}, r_slice: {r_slice.shape}")

                ss_x = self.softsort(x_slice) # B x N x N
                ss_r = self.softsort(r_slice) # B x N_ref x Nref
                #print(f"[slice {l}] ss_x: {ss_x.shape}, ss_r: {ss_r.shape}")
                if N_ref == N:
                    # If the number of reference points equals the number of samples, we can use a direct transport plan                
                    plan = torch.matmul(ss_r, ss_x.transpose(-1, -2)) / N # B x N x N, B different transport plans; 1 per sample
                else:
                    # ss_x: B x N x N       (soft permutation for target samples)
                    # ss_r: B x N_ref x N_ref (soft permutation for reference)
                    # P_interp: N_ref x N (transport plan from reference to target in sorted space)

                    # Expand P_interp for batch
                    P_expanded = P_interp.unsqueeze(0).expand(B, -1, -1)   # B x N_ref x N

                    # Compute approximate transport plan
                    # ss_r: B x N_ref x N_ref  (unsorts reference)
                    # P_expanded: B x N_ref x N  (maps reference to target in sorted space)
                    # ss_x: B x N x N  (unsorts target)
                    # 
                    # The idea:
                    #   1. Apply ss_r to "unsort" reference indices
                    #   2. Apply ss_x^T to "unsort" target indices
                    #
                    plan = torch.matmul(ss_r, P_expanded)                   # B x N_ref x N
                    plan = torch.matmul(plan, ss_x.transpose(-1, -2))       # B x N_ref x N


                dist_l = torch.mean((cost * plan).sum(dim=(-1, -2)))
                per_slice_distances.append(dist_l)

            per_slice_distances = torch.stack(per_slice_distances)#.to(X.device)

            return embeddings, per_slice_distances
        
        else:
            return embeddings, torch.zeros(self.num_projections).to("cuda")

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)