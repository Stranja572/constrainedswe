#!/usr/bin/env python
import argparse, random
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


# Match B's idea of "constraint-based" poolings
CONSTRAINED_POOLINGS = {"SWE", "CSWE", "CONSTRAINED_SWE", "CONSTRAINED_ESP"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--pooling", type=str, default="CSWE")
    p.add_argument("--projections", type=int, default=64)
    p.add_argument("--npieces", type=int, default=64)
    p.add_argument("--tau_aggregation", type=float, default=0.01,
                   help="Tau aggregation parameter for SWE / CSWE")
    p.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eps_fraction", type=lambda x: float(Fraction(x)), default=0.5)
    p.add_argument("--parallelized", type=lambda x: x == "True", default=True)

    # Keep A's learning rate knobs, but don't require them (B hardcodes LR)
    p.add_argument("--primal_lr", type=float, default=1e-3)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.5)

    # If you want A to mirror B's EPS computation, pass in the baseline.
    # If you already compute EPS elsewhere, set --unconstrained_violation accordingly or just pass --epsilon directly.
    p.add_argument("--unconstrained_violation", type=float, default=None,
                   help="Baseline raw violation used to compute EPS = eps_fraction * unconstrained_violation.")
    p.add_argument("--epsilon", type=float, default=None,
                   help="If set, overrides EPS computed from eps_fraction.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(args, model, train_loader, val_loader, test_loader):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print hyperparameters like B
    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    pooling_name = args.pooling.upper()
    is_constrained = pooling_name in CONSTRAINED_POOLINGS

    # EPS handling (B does EPS = eps_fraction * UNCONSTRAINED_VIOLATION)
    if args.epsilon is not None:
        EPS = float(args.epsilon)
    elif args.unconstrained_violation is not None:
        EPS = float(args.eps_fraction) * float(args.unconstrained_violation)
    else:
        # If neither is provided, default to 0.0 so "dists - EPS" is still defined.
        EPS = 0.0
    print(f"EPS: {EPS}")

    # DDP unwrap like A
    model = model.module if hasattr(model, "module") else model
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer: keep A’s intent (pool+classifier if constrained-style, else classifier-only)
    params = [p for p in model.parameters() if p.requires_grad]


    optim = Adam(params, lr=args.primal_lr)
    scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)

    phases = ["train", "valid", "test"]
    loaders = {"train": train_loader, "valid": val_loader, "test": test_loader}

    best_val_acc = 0.0
    best_epoch = -1
    best_train_acc1 = best_val_acc1 = best_test_acc1 = 0.0

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Per-epoch phase metrics
        epoch_acc1 = dict.fromkeys(phases, 0.0)
        epoch_loss = dict.fromkeys(phases, 0.0)
        epoch_violation = dict.fromkeys(phases, 0.0)
        epoch_raw_violation = dict.fromkeys(phases, 0.0)

        val_improved = False

        for phase in phases:
            is_train = phase == "train"
            if is_train:
                tic = time.time()
                model.train()
            else:
                model.eval()

            run_loss = 0.0
            run_acc1 = 0.0
            run_viol = 0.0
            run_raw_viol = 0.0

            for x, y in loaders[phase]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    out = model(x)

                    logits = None
                    dists = None
                    lambdas = None

                    # Accept common return shapes
                    if isinstance(out, (tuple, list)):
                        if len(out) == 2:
                            logits, dists = out
                        elif len(out) >= 3:
                            logits, dists, lambdas = out[0], out[1], out[2]
                        else:
                            logits = out[0]
                    else:
                        logits = out

                    loss = criterion(logits, y)

                    if is_train and is_constrained:
                        # If lambdas not returned, try model.lambdas
                        if lambdas is None:
                            if hasattr(model, "lambdas"):
                                lambdas = model.lambdas
                            else:
                                raise RuntimeError(
                                    "Constrained pooling requested but no lambdas provided "
                                    "(neither returned by model(x) nor found at model.lambdas)."
                                )
                        if dists is None:
                            raise RuntimeError(
                                "Constrained pooling requested but no dists/raw violations returned by model(x)."
                            )

                        raw_violation = dists
                        constraint_violation = dists - EPS

                        lagrangian = loss + torch.sum(lambdas * (dists - EPS))
                        lagrangian.backward()
                        optim.step()
                    else:
                        raw_violation = torch.tensor(0.0, device=device)
                        constraint_violation = torch.tensor(0.0, device=device)

                        if is_train:
                            loss.backward()
                            optim.step()

                # Metrics
                bs = x.size(0)
                top1 = (logits.argmax(1) == y).float().mean().item()

                run_loss += loss.item() * bs
                run_acc1 += top1 * bs

                # B prints violations only for TRAIN and only for constrained poolings
                if is_train and is_constrained:
                    run_raw_viol += raw_violation.mean().item() * bs
                    run_viol += constraint_violation.mean().item() * bs

            N = len(loaders[phase].dataset)
            epoch_loss[phase] = run_loss / N
            epoch_acc1[phase] = run_acc1 / N
            epoch_raw_violation[phase] = run_raw_viol / N
            epoch_violation[phase] = run_viol / N

            # Prints exactly in B's style
            name = phase.capitalize()
            msg = (f"{name:5s} Epoch {epoch:3d} | "
                   f"loss {epoch_loss[phase]:.4f} | "
                   f"Acc@1 {epoch_acc1[phase]:.4f}")
            if phase == "train" and is_constrained:
                msg += (f" | mean raw viol {epoch_raw_violation[phase]:.4f}"
                        f" | mean viol {epoch_violation[phase]:.4f}")
            if phase == "train":
                msg += f" | time {time.time()-tic:.2f}s"
            print(msg)

        scheduler.step()

        # Keep best by valid Acc@1 (B behavior)
        if epoch_acc1["valid"] > best_val_acc:
            val_improved = True
            best_val_acc = epoch_acc1["valid"]
            best_epoch = epoch
            best_val_acc1 = epoch_acc1["valid"]
            best_train_acc1 = epoch_acc1["train"]

        if val_improved:
            best_test_acc1 = epoch_acc1["test"]

    print("\n========== BEST MODEL ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Acc@1: {best_train_acc1:.4f}")
    print(f"Valid Acc@1: {best_val_acc1:.4f}")
    print(f"Test  Acc@1: {best_test_acc1:.4f}")
    print("================================\n")

