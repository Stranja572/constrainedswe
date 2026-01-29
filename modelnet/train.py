#!/usr/bin/env python
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
from fractions import Fraction

from pct import NaivePCT, NaivePCT_cls
from ModelNet40_data import ModelNet40

#Pooling methods
#d_in = 256, pooling 512 vectors
from cswe import SWE_Pooling
from swe import SWE_PoolingRun
from fswlib import FSWEmbedding

# --------------------------------------------------------------------------- #
# Fixed hyper‑parameters (no sweep)                                           #
# --------------------------------------------------------------------------- #
BACKBONE             = "PCT"
NUM_POINTS_PER_SET   = 512 #number of reference points / points per point cloud
DUAL_LR              = 1e-3

ALPHA_LAPSUM = 10

BASE_RANDOM_SEED     = 42
OUTPUT_DIM           = 256
LR                   = 1e-3
NUM_CLASSES          = 40

UNCONSTRAINED_VIOLATION = 18.665

# --------------------------------------------------------------------------- #
# CLI & reproducibility                                                       #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int, default=BASE_RANDOM_SEED,
                   help="Random seed (supplied by the Slurm array).")
    p.add_argument("--pooling", type=str, default="CSWE")
    p.add_argument("--projections", type=int, default=64)
    p.add_argument("--npieces", type=int, default=16)
    p.add_argument( "--tau_aggregation",  type=float,  default=0.01,  help="Tau aggregation parameter for SWE / CSWE")
    p.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eps_fraction", type=lambda x: float(Fraction(x)), default = 0.5)
    p.add_argument("--parallelized", type=lambda x: x == "True", default=True)
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
    def __init__(self, pooling_name, d_in, num_slices, num_ref_points, npieces=64, alpha_lapsum=10, dual_lr=1e-3, eps=10000, tau_aggregation=0.01, parallelized = False):
        super().__init__()
        self.pooling_name = pooling_name

        if pooling_name.upper() in {"CSWE", "CONSTRAINED_SWE"}:
            # typical flattened size: (B, num_ref_points * num_slices)
            self.output_dim = num_ref_points + num_slices
            self.pool = SWE_Pooling(
                d_in=d_in,
                num_slices=num_slices,
                num_ref_points=num_ref_points,
                alpha_lapsum=alpha_lapsum,
                dual_lr=dual_lr,
                eps=eps,
                tau_aggregation=tau_aggregation,
                parallelized = parallelized
            )

        elif pooling_name.upper() == "SWE":
            self.output_dim = num_ref_points + num_slices
            self.pool = SWE_Pooling(
                d_in=d_in,
                num_slices=num_slices,
                num_ref_points=num_ref_points,
                alpha_lapsum=alpha_lapsum,
                dual_lr=dual_lr,
                eps=10000,
                tau_aggregation=tau_aggregation,
                parallelized = parallelized
            )
            # self.pool= SWE_PoolingRun(
            #     d_in = d_in, num_ref_points = num_ref_points, num_projections=num_slices
            # )


        elif self.pooling_name.upper() == "GAP":
            self.output_dim = d_in
        
            
        
        elif self.pooling_name.upper() == "FSW":
            self.output_dim = num_ref_points + num_slices 
            self.pool = FSWEmbedding(d_in=d_in, d_out=self.output_dim, device='cuda')
    

        else:
            raise ValueError(f"Pooling {pooling_name} not implemented")
        
        print(f"Pooling output dimension: {self.output_dim}")

    def forward(self, P):
        if self.pooling_name == "GAP":
            return torch.mean(P, dim=1), None
      
        # elif self.pooling_name.upper() == "SWE":
        #     pooled, _, _ = self.pool(P)                # B × M + L
        #     return pooled.reshape(-1, self.output_dim), dists , lambdas
        elif self.pooling_name.upper() == "FSW":
            pooled = self.pool(P)                
            return pooled.reshape(-1, self.output_dim), None 
        # elif self.pooling_name.upper() == "SWE": #Do this for SWEPoolingRun
        #     pooled = self.pool(P)       
        #     return pooled.reshape(-1, self.output_dim), None #B x (M + L)
 
        else: #SWE, CSWE
            pooled, dists, lambdas = self.pool(P)
            return pooled.reshape(-1, self.output_dim), dists, lambdas

# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_test(args):
    set_seed(args.seed)
    device = torch.device("cuda:0")

    NUM_PROJECTIONS      = args.projections
    EPS = 10000

    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if args.pooling == "constrained_swe" or args.pooling == "CSWE":
        EPS = args.eps_fraction*UNCONSTRAINED_VIOLATION #18.665 * __
    print(f"EPS: {EPS}")
    
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
    loader  = {p: DataLoader(dataset[p], batch_size=args.batch_size, shuffle=(p=="train"))
               for p in phases}

    # Model
    backbone = Backbone(BACKBONE, OUTPUT_DIM)

    #backbone.load_state_dict(torch.load("./ckpts/backbone.pt"))

    
    state_dict = torch.load("./ckpts/backbone.pt", map_location="cuda:0", weights_only=False) #make sure weights go to valid gpu 
    backbone.load_state_dict(state_dict)
    for p in backbone.parameters():
        p.requires_grad = False
    
    for name, param in backbone.named_parameters():
        param.requires_grad = False
    backbone.eval()

    pooling = Pooling(
    args.pooling,
    OUTPUT_DIM,
    args.projections,
    NUM_POINTS_PER_SET,
    npieces=args.npieces,
    alpha_lapsum=ALPHA_LAPSUM,
    dual_lr=DUAL_LR,
    eps=EPS,
    tau_aggregation=args.tau_aggregation,
    parallelized = args.parallelized
)

    classifier = nn.Linear(pooling.output_dim, NUM_CLASSES)

    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    params = [p for p in pooling.parameters() if p.requires_grad] + list(classifier.parameters()) #Just classifier parameters if gap

    criterion = nn.CrossEntropyLoss()
    optim     = Adam(params, lr=LR)
    scheduler = StepLR(optim, step_size=50, gamma=0.5)

    epoch_metrics, best_val_acc = defaultdict(list), 0.0
    best_epoch = -1
    # Train / Val / Test metrics (top‑1) captured at best val‑epoch
    best_train_acc1 = best_val_acc1 = best_test_acc1 = 0.0

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        val_improved = False
        # Per‑epoch snapshots so we can tie metrics from the same epoch together
        epoch_acc1       = dict.fromkeys(phases, 0.0)
        epoch_loss       = dict.fromkeys(phases, 0.0)
        epoch_violation  = dict.fromkeys(phases, 0.0)   # CSWE / C‑ESP only
        epoch_raw_violation = dict.fromkeys(phases, 0.0)


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
            run_raw_viol = 0.0


            for batch_idx, (x, y) in enumerate(loader[phase]):

                if phase == "test" and batch_idx == BATCH_TO_TIME:
                    torch.cuda.synchronize() #barrier
                    batch_start = time.time()

                x = x.to(device).float()
                y = y.to(device).squeeze()
                optim.zero_grad()

                dists = None
                lambdas = None

                with torch.set_grad_enabled(is_train):
                    z = backbone(x)
                    if args.pooling.upper() == "FSPOOL":
                        z = z.swapdims(1, 2) #FSPool takes b x d_in x n
                        v, dists = pooling(z)
                    elif args.pooling.upper() == "CLS": #save some time
                        v = z #v is just the CLS token for CLS
                    elif args.pooling.upper() in {"FSW", "GAP", "SWE"}: 
                        v, dists = pooling(z)
                    else:
                        v, dists, lambdas = pooling(z)

                    if is_train and args.pooling.upper() in {"CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
                        raw_violation = dists
                        constraint_violation = dists - EPS
                    else: #SWE
                        raw_violation = torch.tensor(0.0, device=device)
                        constraint_violation = torch.tensor(0.0, device=device)
                    
                    logits   = classifier(v)
                    loss     = criterion(logits, y)

                
                    # ----- dual updates (CSWE / C‑ESP) -----
                    
                    if is_train and args.pooling.upper() in {"CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
                        lagrangian = loss + torch.sum(lambdas * (dists - EPS))
                        lagrangian.backward()
                        optim.step()
                    else: #SWE
                        if is_train:
                            loss.backward()
                            optim.step()

                # ----- metrics -----
                bs = x.size(0)
                top1 = (logits.argmax(1) == y).float().mean().item()
                run_loss   += loss.item() * bs
                run_acc1   += top1 * bs
                if is_train:
                    run_viol   += constraint_violation.mean().item() * bs
                    run_raw_viol += raw_violation.mean().item() * bs

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
            epoch_raw_violation[phase] = run_raw_viol / N

            # ---- prints ----
            name = phase.capitalize()
            msg  = (f"{name:5s} Epoch {epoch:3d} | "
                    f"loss {epoch_loss[phase]:.4f} | "
                    f"Acc@1 {epoch_acc1[phase]:.4f}")
            if phase == "train" and args.pooling.upper() in {"SWE", "CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}:
                msg += (f" | mean raw viol {epoch_raw_violation[phase]:.4f}"
                        f" | mean viol {epoch_violation[phase]:.4f}")
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
