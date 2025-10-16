#!/usr/bin/env python
"""
train.py — ModelNet40 experiment with either CSWE or GAP pooling.

Examples
--------
CSWE (constrained):   python train.py --seed 42           # default pooling=CSWE
GAP (avg‑pool only):  python train.py --seed 42 --pooling GAP
"""

# --------------------------------------------------------------------------- #
# Imports                                                                     #
# --------------------------------------------------------------------------- #
import argparse, os, random, time
from collections import defaultdict


import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools


from pct import NaivePCT, NaivePCT_cls
from ModelNet40_data import ModelNet40

#Pooling methods
#d_in = 256, pooling 512 vectors
from constrained import ConstrainedSWE
from constrainedesp import ConstrainedExpectedSlicedPlan
from swe import SWE_Pooling
from lot import LOTSinkhorn
from fswlib import FSWEmbedding
from fs import FSPool
from covariance import covariance_pool

# --------------------------------------------------------------------------- #
# Fixed hyper‑parameters (no sweep)                                           #
# --------------------------------------------------------------------------- #
BACKBONE             = "PCT"
NUM_POINTS_PER_SET   = 512 #number of reference points
ALPHA_SLACK          = 1
DUAL_LR              = 1e-3
EPS                  = 7 #maybe eps is different for constrained_esp
TAU_SOFTSORT         = 1e-3

BATCH_SIZE           = 64
BASE_RANDOM_SEED     = 42
NUM_EPOCHS           = 200
OUTPUT_DIM           = 256
LR                   = 1e-3
NUM_CLASSES          = 40

# --------------------------------------------------------------------------- #
# CLI & reproducibility                                                       #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int, default=BASE_RANDOM_SEED,
                   help="Random seed (supplied by the Slurm array).")
    p.add_argument("--pooling", type=str, default="CSWE")
    p.add_argument("--projections", type=int, default=64)
    p.add_argument("--npieces", type=int, default=64)
    return p.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# --------------------------------------------------------------------------- #
# Model components                                                            #
# --------------------------------------------------------------------------- #
class Backbone(nn.Module):
    def __init__(self, backbone_type: str, output_dim: int):
        super().__init__()
        if backbone_type == "PCT":
            self.backbone = NaivePCT(d_out=output_dim)
        elif backbone_type == "PCT_CLS":
            self.backbone = NaivePCT_cls(d_out=output_dim)
        else:
            raise ValueError(f"Backbone {backbone_type} not implemented")

    def forward(self, x):
        return self.backbone(x)


class Pooling(nn.Module):
    """CSWE (with constraints) or GAP (plain average)."""
    def __init__(self, pooling_name, d_in, num_projections, num_ref_points, tau_softsort, npieces = 64):
        super().__init__()
        self.pooling_name = pooling_name

        if pooling_name == "constrained_swe":
            self.output_dim = num_ref_points * num_projections
            self.register_buffer("lambdas", torch.zeros(num_projections))
            self.slacks = nn.Parameter(torch.zeros(num_projections))
            self.pool = ConstrainedSWE(
                d_in=d_in,
                num_ref_points=num_ref_points,
                num_projections=num_projections,
                tau_softsort=tau_softsort,
            )

        elif self.pooling_name.upper() == "SWE":
            self.output_dim = num_ref_points * num_projections
            self.pool = SWE_Pooling(
                d_in=d_in,
                num_ref_points=num_ref_points,
                num_projections=num_projections
            )

        elif self.pooling_name.upper() == "GAP":
            self.output_dim = d_in
            #self.register_buffer("lambdas", torch.zeros(1))
            #self.slacks = nn.Parameter(torch.zeros(1), requires_grad=False) #use this later on to ignore slacks for GAP
        
        elif self.pooling_name.upper() == "CLS":
            self.output_dim = d_in


        elif self.pooling_name.upper() in {"CONSTRAINED_ESP", "CESP"}:
            #  C‑ESP:  (B , N_ref , d_in )  →  flatten to (B , N_ref·d_in)
            self.output_dim = num_ref_points * d_in

            # dual variables for the slice‑wise constraints – exactly the
            # same shape & training logic you already use for CSWE
            self.register_buffer("lambdas", torch.zeros(num_projections))
            self.slacks  = nn.Parameter(torch.zeros(num_projections))

            self.pool = ConstrainedExpectedSlicedPlan(
                d_in=d_in,
                num_ref_points=num_ref_points,
                num_projections=num_projections,
                tau_softsort=tau_softsort,
            )
        
        elif self.pooling_name.upper() == "LOT":
            # LOT produces a (B, N_ref, d_in) tensor, same as C‑ESP      
            self.output_dim = num_ref_points * d_in
            self.pool = LOTSinkhorn(
                d_in=d_in,
                num_ref_points=num_ref_points
            )
        
        elif self.pooling_name.upper() == "FSW":
            self.output_dim = num_ref_points * 64
            self.pool = FSWEmbedding(d_in=d_in, d_out=self.output_dim, device='cuda')
        
        elif self.pooling_name.upper() == "FSPOOL": # (N x d) -> (d)
            self.output_dim = d_in
            self.pool = FSPool(in_channels=d_in, n_pieces=npieces) 

        elif pooling_name.upper() == "COVARIANCE":
            self.output_dim = d_in + (d_in*(d_in+1))//2 
            self.pool = covariance_pool

        else:
            raise ValueError(f"Pooling {pooling_name} not implemented")
        
        print(f"Pooling output dimension: {self.output_dim}")

    def forward(self, P):
        if self.pooling_name == "GAP":
            return torch.mean(P, dim=1), None
        elif self.pooling_name.upper() == "CLS":
            return P, None
        elif self.pooling_name.upper() == "LOT":
            pooled = self.pool(P)                
            return pooled.reshape(-1, self.output_dim), None
        elif self.pooling_name.upper() == "SWE":
            pooled = self.pool(P)                # B × ML
            return pooled.reshape(-1, self.output_dim), None
        elif self.pooling_name.upper() == "FSW":
            pooled = self.pool(P)                
            return pooled.reshape(-1, self.output_dim), None
        elif self.pooling_name.upper() == "FSPOOL":
            pooled, __ = self.pool(P)             
            return pooled.reshape(-1, self.output_dim), None
        elif self.pooling_name.upper() == "COVARIANCE":
            pooled = self.pool(P)             
            return pooled.reshape(-1, self.output_dim), None

        else: #CSWE or CESP
            pooled, dists = self.pool(P)
            return pooled.reshape(-1, self.output_dim), dists

# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_test(args):
    set_seed(args.seed)
    device = torch.device("cuda:0")

    NUM_PROJECTIONS      = args.projections

    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if args.pooling == "SWE":
        global EPS
        EPS = 10000
    
    if args.pooling == "cls":
        global BACKBONE
        BACKBONE = "PCT_CLS"

    results_dir = (
        f"./results/modelnet40_{args.pooling.lower()}/"
        f"{BACKBONE}_{args.pooling}_proj{NUM_PROJECTIONS}_seed{args.seed}/"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Data
    phases  = ["train", "valid", "test"]
    dataset = {p: ModelNet40(NUM_POINTS_PER_SET, partition=p) for p in phases}
    loader  = {p: DataLoader(dataset[p], batch_size=BATCH_SIZE, shuffle=(p=="train"))
               for p in phases}

    # Model
    backbone = Backbone(BACKBONE, OUTPUT_DIM)

    #backbone.load_state_dict(torch.load("./ckpts/backbone.pt"))

    if BACKBONE == 'PCT_CLS':
        state_dict = torch.load("./ckpts/backbone.pt", map_location="cuda:0", weights_only=True) #make sure weights go to valid gpu 

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        backbone.load_state_dict(state_dict, strict=False)

        
        
    else:
        state_dict = torch.load("./ckpts/backbone.pt", map_location="cuda:0", weights_only=True) #make sure weights go to valid gpu 
        backbone.load_state_dict(state_dict)
        for p in backbone.parameters():
            p.requires_grad = False
    
    for name, param in backbone.named_parameters():
            #print(name)
            if name != 'cls_token':
                param.requires_grad = False

    if BACKBONE == 'PCT_CLS':
            backbone.backbone.cls_token.requires_grad = True
    backbone.eval()

    pooling = Pooling(args.pooling, OUTPUT_DIM, NUM_PROJECTIONS, NUM_POINTS_PER_SET, TAU_SOFTSORT, args.npieces)

    classifier = nn.Linear(pooling.output_dim, NUM_CLASSES)

    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    if args.pooling.upper() in {"CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
        lambdas = pooling.lambdas
        slacks  = pooling.slacks
    else:
        lambdas = slacks = None

    

    params = [p for p in pooling.parameters() if p.requires_grad] + list(classifier.parameters()) #Just classifier parameters if gap

    criterion = nn.CrossEntropyLoss()
    optim     = Adam(params, lr=LR)
    scheduler = StepLR(optim, step_size=50, gamma=0.5)

    epoch_metrics, best_val_acc = defaultdict(list), 0.0
    best_epoch = -1
    # Train / Val / Test metrics (top‑1) captured at best val‑epoch
    best_train_acc1 = best_val_acc1 = best_test_acc1 = 0.0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        val_improved = False
        # Per‑epoch snapshots so we can tie metrics from the same epoch together
        epoch_acc1       = dict.fromkeys(phases, 0.0)
        epoch_acc5       = dict.fromkeys(phases, 0.0)
        epoch_loss       = dict.fromkeys(phases, 0.0)
        epoch_violation  = dict.fromkeys(phases, 0.0)   # CSWE / C‑ESP only

        for phase in phases:
            is_train = phase == "train"
            BATCH_TO_TIME = 1 #time of the second mini batch
            timed = False

            if is_train:
                tic = time.time()
                pooling.train(); classifier.train()
            else:
                pooling.eval(); classifier.eval()

            # Running totals
            run_loss = run_acc1 = 0
            run_viol = 0.0

            for batch_idx, (x, y) in enumerate(loader[phase]):

                if phase == "test" and batch_idx == BATCH_TO_TIME:
                    torch.cuda.synchronize() #barrier
                    batch_start = time.time()

                x = x.to(device).float()
                y = y.to(device).squeeze()
                optim.zero_grad()

                with torch.set_grad_enabled(is_train):
                    z = backbone(x)
                    if args.pooling.upper() == "FSPOOL":
                        z = z.swapdims(1, 2) #FSPool takes b x d_in x n
                        v, dists = pooling(z)
                    elif args.pooling.upper() == "CLS": #save some time
                        v = z
                    else:
                        v, dists = pooling(z) #v is just the CLS token for CLS
                    
                    logits   = classifier(v)
                    loss     = criterion(logits, y)

                    # ----- dual updates (CSWE / C‑ESP) -----
                    if args.pooling.upper() in {"CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
                        constraint_violation = dists - (EPS + slacks)
                        lagrangian = (
                            loss
                            + torch.sum(lambdas * constraint_violation)
                            + 0.5 * ALPHA_SLACK * torch.linalg.norm(slacks) ** 2
                        )
                        if is_train:
                            lagrangian.backward()
                            optim.step()
                            lambdas += DUAL_LR * constraint_violation.detach()
                            lambdas.clamp_(min=0); slacks.data.clamp_(min=0)
                    else:  
                        constraint_violation = torch.tensor(0.0, device=device)
                        if is_train:
                            loss.backward(); optim.step()

                # ----- metrics -----
                bs = x.size(0)
                top1 = (logits.argmax(1) == y).float().mean().item()
                run_loss   += loss.item() * bs
                run_acc1   += top1 * bs
                run_viol   += constraint_violation.mean().item() * bs

                if phase == "test" and batch_idx == BATCH_TO_TIME and not timed:
                    torch.cuda.synchronize() #for correct time
                    elapsed = time.time() - batch_start
                    print(f"[Epoch {epoch}] Inference time for first test mini‑batch: {elapsed:.4f}s")
                    timed = True

            # ---- epoch aggregates ----
            N = len(loader[phase].dataset)
            epoch_loss[phase]      = run_loss / N
            epoch_acc1[phase]      = run_acc1 / N
            epoch_violation[phase] = run_viol / N

            # ---- prints ----
            name = phase.capitalize()
            msg  = (f"{name:5s} Epoch {epoch:3d} | "
                    f"loss {epoch_loss[phase]:.4f} | "
                    f"Acc@1 {epoch_acc1[phase]:.4f}")
            if args.pooling.upper() in {"CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
                msg += f" | mean viol {epoch_violation[phase]:.4f}"
            if phase == "train":
                msg += f" | time {time.time()-tic:.2f}s"
            print(msg)

        # ---- scheduler step ----
        scheduler.step()

        # ---- keep best (highest val Acc@1) ----
        if epoch_acc1["valid"] > best_val_acc:
            val_improved = True
            best_val_acc   = epoch_acc1["valid"]
            best_epoch     = epoch
            best_val_acc1  = epoch_acc1["valid"]
            best_train_acc1 = epoch_acc1["train"]
        
        if val_improved == True and phase == "test":
            best_test_acc1 = epoch_acc1["test"]


    print("\n========== BEST MODEL ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Acc@1: {best_train_acc1:.4f}")
    print(f"Valid Acc@1: {best_val_acc1:.4f}")
    print(f"Test  Acc@1: {best_test_acc1:.4f}")
    print("================================\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train_test(parse_args())
