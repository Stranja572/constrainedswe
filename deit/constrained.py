import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchinterp1d import Interp1d

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

class ConstrainedSWE(nn.Module):
    def __init__(self, d_in, num_ref_points, num_projections, tau_softsort=1, embedding = "flatten", parallel = True):
        '''
        The PSWE module that produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_projections: Number of slices
        '''
        super(ConstrainedSWE, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_projections = num_projections

        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, d_in) #m x d_in (reference points are in the original space here)
        self.reference = nn.Parameter(uniform_ref) 

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

        self.map = None

        self.parallel = parallel

        self.embedding = embedding

        if self.embedding == "mapM":
            self.map = nn.Linear(num_projections,1)
        elif self.embedding == "mapL":
            self.map = nn.Linear(num_ref_points,1) #L x M -> L

    def forward(self, X):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: B x num_projections tensor, containing a batch of B embeddings, each of dimension "num_projections" (i.e., number of slices)
        '''

        B, N, dn = X.shape
        Xslices = self.get_slice(X)
        Xslices_sorted, Xind = torch.sort(Xslices, dim=1)

        M, dm = self.reference.shape

        if M == N:
            Xslices_sorted_interpolated = Xslices_sorted
        else:
            x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_projections, -1)
            Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_projections, -1), 1, 2)

        # Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)
        Rslices = self.get_slice(self.reference).expand(Xslices_sorted_interpolated.shape) #We also slice the reference here

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1) #B x L x M

        if self.embedding == "flatten":
            embeddings = embeddings.reshape(B, -1) #B x LM; view requires contiguous
        elif self.embedding == "mapM":
            embeddings = embeddings.transpose(2,1) #B x M x L; transpose arguments are symmetric it just swaps the dimensions 
            embeddings = self.map(embeddings) #B x M x 1
            embeddings = embeddings.squeeze(2) #B x M
        elif self.embedding == "mapL":
            # embeddings = self.map(embeddings) #B x L x 1
            # embeddings = embeddings.squeeze(2) #B x L

            #Possibly faster? GPU not optimized for M->1 (skinny linear layer matrix is bad)
            w = self.map.weight.view(-1)              # shape: (M,)
            b = self.map.bias                          # shape: (1,)
            embeddings = torch.einsum('blm,m->bl', embeddings, w) + b 

        elif self.embedding == "meanL": #PyTorch automatically gets rid of the singleton dimension for mean
            embeddings = embeddings.mean(dim=2) #mean along dimension M; B x L
        elif self.embedding == "meanM":
            embeddings = embeddings.mean(dim=1) #mean along dimension L; B x M
        
        sliced_X = Xslices_sorted_interpolated if M != N else Xslices_sorted
        ref_expanded = self.reference.unsqueeze(0).repeat(B, 1, 1)     # [B×M×D] vs B x N x D
        cost         = torch.cdist(X, ref_expanded, p=2)              # [B×N×N], when M=N?

        if self.parallel == False:

            per_slice_distances = []

            #For one slice, our dist_l is the mean of the SWGG cost going from reference slice distribution to the N token distribution (of one sample) for all samples
            #Basically we are minimizing the average SWGG per slice (across a mini batch), not looking at it per sample
            for l in range(self.num_projections):
                x_slice = sliced_X[:, :, l] # B×N
                r_slice = Rslices[:, :, l] # B×M
                #print(f"[slice {l}] x_slice: {x_slice.shape}, r_slice: {r_slice.shape}")

                ss_x = self.softsort(x_slice)
                ss_r = self.softsort(r_slice)
                #print(f"[slice {l}] ss_x: {ss_x.shape}, ss_r: {ss_r.shape}")

                plan = torch.matmul(ss_x.transpose(-1, -2), ss_r) / N # B x N x N, B different transport plans; 1 per sample

                dist_l = torch.mean((cost * plan).sum(dim=(-1, -2)))

                per_slice_distances.append(dist_l)


            per_slice_distances = torch.stack(per_slice_distances)#.to(X.device)
        
        else: #Parallel = true
            
            L      = self.num_projections

            #Flatten slices to one big batch of size B·L
            x_flat = sliced_X.permute(0,2,1).reshape(B*L, N)  # [B·L×N]
            r_flat = Rslices.permute(0,2,1).reshape(B*L, M)   # [B·L×N]

            #Softsort
            ss_x_flat = self.softsort(x_flat)  # [B·L×N×N]
            ss_r_flat = self.softsort(r_flat)  # [B·L×NxN]

            #BL transport plans (B per slice); [B·L×N×N] * [B·L×N×N] -> [B·L×N×N]
            plans_flat = torch.matmul(ss_x_flat.transpose(1,2), ss_r_flat) / N

            #Expand cost and compute SWGG per slice averaged across batch
            cost_exp     = cost.unsqueeze(1).expand(-1, L, -1, -1)        # [B×L×N×N]
            cost_flat    = cost_exp.reshape(B*L, N, N)                   # [B·L×N×N]

            dist_flat    = (cost_flat * plans_flat).sum(dim=(-1,-2))     # [B·L]
            per_slice_distances    = dist_flat.view(B, L).mean(dim=0)    # [L]

        return embeddings, per_slice_distances


    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)

