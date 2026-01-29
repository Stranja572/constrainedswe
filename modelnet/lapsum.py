import numpy as np
import torch


class SoftTopK(torch.autograd.Function):
    # @staticmethod
    # def _solve(s, t, a, b, e):
    #     z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
    #     ab = torch.where(e > 0, a, b)

    #     return torch.where(
    #         e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
    #     )
    
    ## A more stable version of _solve
    @staticmethod
    def _solve(s, t, a, b, e, eps=1e-12):
        s, t, a, b, e = [arg.to(torch.float64) for arg in (s, t, a, b, e)]

        diff = torch.clamp(s - t, max=40.0)           # avoid exp overflow
        exp_term = torch.exp(diff)

        prod = torch.clamp(a * b * exp_term, min=0.0, max=1e30)
        inside = torch.clamp(e**2 + prod, min=0.0, max=1e30)
        sqrt_term = torch.sqrt(inside)

        z = torch.clamp(torch.abs(e) + sqrt_term, min=eps)
        ab = torch.clamp(torch.where(e > 0, a, b), min=eps)

        out_pos = t + torch.log(z) - torch.log(ab)
        out_neg = s - torch.log(z) + torch.log(ab)

        out = torch.where(e > 0, out_pos, out_neg)
        out = torch.clamp(out, -1e6, 1e6)             # keep finite
        return out.to(s.dtype)


    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            # eA.add_(scaled).exp_()
            # safer exponential
            tmp = eA + scaled
            tmp = torch.clamp(tmp, max=40.0)     # exp(40) ≈ 2e17, large but finite
            eA = torch.exp(tmp)

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)

            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)

            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)

            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()
        # print(b.max().item(), b.min().item())
        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)

        torch.where(sign_x, 1 - p, p, out=p)

        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    # @staticmethod
    # def backward(ctx, grad_output):
    #     r, x, S = ctx.saved_tensors
    #     alpha = ctx.alpha

    #     q = torch.softmax(-torch.abs(x), dim=1)
    #     qgrad = q * grad_output

    #     # Gradients
    #     grad_k=qgrad.sum(dim=1)
    #     grad_r = S * q * (grad_k.unsqueeze(1)-grad_output)
    #     grad_alpha = (S / alpha * qgrad * (r - (q * r).sum(dim=1, keepdim=True))).sum()
    #     return grad_r, grad_k, grad_alpha, None

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha

        x.abs_().neg_()
        
        # print("before softmax:", x.min().item(), x.max().item())
        q = torch.softmax(x, dim=1)
        # if torch.isnan(q).any() or torch.isinf(q).any():
        #     raise ValueError("NaN/Inf in q after softmax")

        torch.mul(q, grad_output, out=x)
        grad_k = x.sum(dim=1, keepdim=True)

        grad_r = grad_k - grad_output
        grad_r.mul_(q).mul_(S)

        q.mul_(r)
        x.mul_(S / alpha)  # grad_alpha = (S / alpha) * x
        #r.sub_(q.sum(dim=1, keepdim=True))
        r = r - q.sum(dim=1, keepdim=True)

        x.mul_(r)  # grad_alpha.mul_(r)
        grad_alpha = x.sum()  # grad_alpha = grad_alpha.sum()
        return grad_r, grad_k.squeeze(1), grad_alpha, None
        # return grad_r, grad_k.squeeze(1), None, None ## For fixed alpha


def soft_top_k(r, k, alpha, descending=False):
    return SoftTopK.apply(r, k, alpha, descending)



def soft_permutation(r, alpha=0.03, descending=False):
    r_ = r.unsqueeze(0)
    k = torch.arange(1, r_.shape[-1]).to(r.device)
    br = r_.repeat((len(k), 1))

    softk = soft_top_k(br, k, alpha, descending=descending)

    result = torch.concat((torch.zeros(1, softk.shape[-1]).to(r.device), softk, torch.ones(1, softk.shape[-1]).to(r.device)), dim=0)

    Pl = result[1:, :] - result[:-1, :]
    return Pl

def soft_permutation_batch(R, alpha=0.03, descending=False):
    """
    R: (B, N)
    return: (B, N, N)
    """
    B, N = R.shape
    device = R.device

    # k = [1, 2, ..., N-1] for all items in batch
    k = torch.arange(1, N, device=device).unsqueeze(0).expand(B, N-1)

    # Expand R so each row gets repeated for all k
    # br: (B*(N-1), N)
    br = R.unsqueeze(1).expand(B, N-1, N).reshape(-1, N)

    # vectorized k: (B*(N-1))
    bk = k.reshape(-1)

    # soft-top-k gives: (B*(N-1), N)
    softk = soft_top_k(br, bk, alpha, descending=descending)

    # reshape back into (B, N-1, N)
    softk = softk.reshape(B, N-1, N)

    # build result = [0; softk; 1] along the "k" dimension
    zeros = torch.zeros(B, 1, N, device=device)
    ones  = torch.ones(B, 1, N, device=device)

    result = torch.cat((zeros, softk, ones), dim=1)  # (B, N+1, N)

    # Take finite differences: P = result[k] - result[k-1]
    P = result[:, 1:, :] - result[:, :-1, :]          # (B, N, N)

    return P