import argparse
import os
import random
import time
from fractions import Fraction

import numpy as np
import torch
import wandb

from train import train_and_evaluate
from utils.dataloader import get_dataloaders
from model import ConstrainedDeiT


CONSTRAINED_POOLINGS = {"SWE", "CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}


def parse_args():
    p = argparse.ArgumentParser()

    # ---- experiment / data ----
    p.add_argument("--dataset", type=str, default="clipart")          # tinyimage or clipart
    p.add_argument("--num_classes", type=int, default=345)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--output_dir", type=str, default="./checkpoints_real")

    # ---- B-style pooling arguments (what you asked A to support) ----
    p.add_argument("--pooling", type=str, default="CSWE")               # CSWE, SWE, GAP, CLS, etc.
    p.add_argument("--projections", type=int, default=64)
    p.add_argument("--npieces", type=int, default=64)
    p.add_argument("--tau_aggregation", type=float, default=0.01)
    p.add_argument("--eps_fraction", type=lambda x: float(Fraction(x)), default=0.93)
    p.add_argument("--parallelized", type=lambda x: x == "True", default=True)

    # ---- DeiT-specific knobs (kept from your original main) ----
    # (If your model uses these for SWE-style pooling internals)
    p.add_argument("--num_ref_points", type=int, default=196)
    p.add_argument("--tau_softsort", type=float, default=1e-2)
    p.add_argument("--layer_stop", type=int, default=11)
    p.add_argument("--parallel", type=lambda x: x.lower() in ["true", "1", "yes"], default=True)

    # ---- optimization (A had primal_lr; B hardcoded LR; keep as CLI) ----
    p.add_argument("--primal_lr", type=float, default=1e-3)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.5)

    # ---- epsilon handling (no slack variables) ----
    # Priority: --epsilon (explicit) else eps_fraction * unconstrained_violation (if provided) else 0.0
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--unconstrained_violation", type=float, default=25.68)

    # ---- wandb ----
    p.add_argument("--wandb", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    p.add_argument("--wandb_key", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_eps(args) -> float:
    pooling = args.pooling.upper()

    # B behavior: if pooling is unconstrained-ish, EPS effectively huge
    if pooling in {"SWE", "CLS"}:
        return 10000.0

    if args.epsilon is not None:
        return float(args.epsilon)

    if args.unconstrained_violation is not None:
        return float(args.eps_fraction) * float(args.unconstrained_violation)

    # safe default (keeps code defined; you can override via --epsilon)
    return 0.0


def main():
    args = parse_args()
    set_seed(args.seed)

    # Dataset-specific class count
    if args.dataset == "clipart":
        args.num_classes = 345

    # Compute EPS once and store back onto args (train loop will read args.epsilon / EPS)
    args.epsilon = resolve_eps(args)

    # Print hyperparameters like B
    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"EPS: {args.epsilon}")

    # Optional W&B (keep, but avoid slack-related anything)
    if args.wandb:
        if not args.wandb_key:
            raise ValueError("--wandb_key is required when --wandb is True")
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="clipart-real",
            entity="constrained-swe",
            name=args.run_name,
            config=vars(args),
        )
        time.sleep(2)

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # Build model: pass pooling name instead of "classification"
    # Map B args -> your model’s expected names where applicable.
    model = ConstrainedDeiT(
        num_classes=args.num_classes,
        num_ref_points=args.num_ref_points,
        num_projections=args.projections,          # B: --projections
        tau_aggregation=args.tau_aggregation,
        classification=args.pooling,               # keep param name, but feed pooling string
        layer_stop=args.layer_stop,
        parallel=args.parallel,
        eps = args.epsilon
    )

    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs — using DataParallel")
        model = torch.nn.DataParallel(model)

    # B-style training loop (same per-epoch prints / BEST MODEL block)
    train_and_evaluate(args, model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
