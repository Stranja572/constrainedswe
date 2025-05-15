import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import wandb
import os

def train_and_evaluate(args, model, train_loader, val_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    #For dataloader
    model = model.module if hasattr(model, 'module') else model

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.classification == "constrained_swe" or args.classification == "swe":
        optimizer = Adam([
            {'params': model.pool.parameters(), 'lr': args.primal_lr},
            {'params': model.classifier.parameters(), 'lr': args.primal_lr},
            {'params': [model.slacks], 'lr': args.slack_lr}
        ])
    else: #cls
       optimizer = Adam(model.classifier.parameters(), lr=args.primal_lr)

    #Lower L -> less weights -> takes shorter to converge -> decrease learning rate more quickly (ignore for fairness)
    step_size=60
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    print(f"Step size: {step_size}")
    if args.wandb:
        wandb.config.update({"step_size": step_size})
    

    # Best metrics
    best_train_acc1 = best_train_acc5 = 0.0
    best_val_acc1 = best_val_acc5 = 0.0
    best_test_acc1 = best_test_acc5 = 0.0

    best_epoch = 0

    
    for epoch in range(1, args.epochs + 1):
        val_improved = False
        epoch_train_acc1 = epoch_train_acc5 = 0.0

        #For each epoch go through train (and step), then evaluate val, and test
        for phase, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            model.train() if phase == 'train' else model.eval()

            running_loss = running_acc1 = running_acc5 = running_cons = running_violations = 0.0

            for x, y in tqdm(loader, desc=f"Epoch {epoch} [{phase}]"): #tqdm writes to stderr by defualt
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if args.classification == "swe" or args.classification == "constrained_swe": #constrainedswe or swe
                        logits, violations = model(x)  #Violations is L prob (for each slice the cost is already averaged over the batch)

                        #Print the mean of this; use for initial epsilon guess for the constrained
                        #print(f"SWGG Cost Average for All Slices (each averaged over batch): {violations.mean().item()}")

                        cons_batch = violations - (args.epsilon + model.slacks) #L (avg violation per slice across all sample)

                    else:
                        logits = model(x) #cls or mean
                    loss_ce = criterion(logits, y) 

                    if phase == 'train' and (args.classification == "swe" or args.classification == "constrained_swe"): #constrainedswe or swe
                        #Since cons batch is violation delta per slice (avg across all samples)
                        lag = loss_ce + torch.sum(model.lambdas * cons_batch) + 0.5 * args.alpha * torch.linalg.norm(model.slacks) ** 2
                        
                        lag.backward()
                        optimizer.step()
                        
                        model.lambdas += args.dual_lr * cons_batch.detach()
                        model.lambdas.clamp_(min=0)
                        model.slacks.data.clamp_(min=0)
                    
                    elif phase == 'train': #cls or mean
                        loss_ce.backward()
                        optimizer.step()

                    # Metrics
                    top1 = (logits.argmax(dim=1) == y).float().mean().item() 
                    _, preds5 = logits.topk(5, dim=1) #indices of top 5 ([B, 5])
                    top5 = (preds5 == y.view(-1, 1)).any(dim=1).float().mean().item() #comparing with [B, 1] tensor to see if the label is in any of those 5 indices; avg for the batch

                bs = x.size(0)

                #each of these is average of the batch so we multiply by batch size so that when we keep adding we can divide for epoch metrics later
                running_loss += loss_ce.item() * bs
                running_acc1 += top1 * bs 
                running_acc5 += top5 * bs

                if args.classification == "swe" or args.classification == "constrained_swe":
                    running_cons += cons_batch.detach().mean().item() * bs
                    running_violations += violations.detach().mean().item() * bs #total vioaltion for a mini batch

            # Epoch metrics for the specific phase
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc1 = running_acc1 / len(loader.dataset)
            epoch_acc5 = running_acc5 / len(loader.dataset)
            
            if args.classification == "swe" or args.classification == "constrained_swe":
                epoch_cons = running_cons / len(loader.dataset)
                epoch_violation_avg = running_violations / len(loader.dataset)
                print(f"Epoch {epoch} SWGG Avg for phase {phase}: {epoch_violation_avg}")
        
                print(f"Epoch {epoch}: {phase} — Top1, {epoch_acc1}, Top5, {epoch_acc5}, epoch_cons {epoch_cons}")
            else:
                print(f"Epoch {epoch}: {phase} — Top1, {epoch_acc1}, Top5, {epoch_acc5}")

            #For SWE/CSWE
            # Save this epoch's train metrics for later snapshot
            if phase == 'train':
                epoch_train_acc1, epoch_train_acc5 = epoch_acc1, epoch_acc5

            # On new best validation, snapshot train & val and save checkpoint
            if phase == 'val' and epoch_acc1 > best_val_acc1:
                val_improved = True
                best_val_acc1, best_val_acc5 = epoch_acc1, epoch_acc5
                best_train_acc1, best_train_acc5 = epoch_train_acc1, epoch_train_acc5
                best_epoch = epoch

            # When validation improved, record test metrics
            if phase == 'test' and val_improved:
                best_test_acc1, best_test_acc5 = epoch_acc1, epoch_acc5

                if args.wandb:
                    wandb.log({
                        "best_train_acc1": best_train_acc1,
                        "best_train_acc5": best_train_acc5,
                        "best_val_acc1"  : best_val_acc1,
                        "best_val_acc5"  : best_val_acc5,
                        "best_test_acc1": best_test_acc1,
                        "best_test_acc5": best_test_acc5
                    }, step=epoch)

           # Per-phase WandB logging
            if args.wandb:
                log_dict = {
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc1":  epoch_acc1,
                    f"{phase}_acc5":  epoch_acc5,
                }
                if args.classification == "swe" or args.classification == "constrained_swe":
                    log_dict.update({
                        f"{phase}_constraint_violation": epoch_cons,
                        f"{phase}_raw_violation":       epoch_violation_avg
                    })
                wandb.log(log_dict, step=epoch)

                # During training also log dual/slack details
                if phase == 'train' and (args.classification == "swe" or args.classification == "constrained_swe"):
                    mean_l = model.lambdas.detach().cpu().mean().item()
                    mean_s = model.slacks.detach().cpu().mean().item()
                    
                    wandb.log({
                        "mean_lambda": mean_l,
                        "mean_slack": mean_s
                    }, step=epoch)
                    for i, (lam, sl) in enumerate(zip(model.lambdas.tolist(), model.slacks.tolist())):
                        wandb.log({
                            f"lambda_{i}": lam,
                            f"slack_{i}": sl
                        }, step=epoch)

        scheduler.step() #Updates learning rate

    # Final best metrics
    print(f"Best Train Acc@1: {best_train_acc1:.4f}, Acc@5: {best_train_acc5:.4f}")
    print(f"Best Val   Acc@1: {best_val_acc1:.4f}, Acc@5: {best_val_acc5:.4f} at Epoch {best_epoch}")
    print(f"Best Test  Acc@1: {best_test_acc1:.4f}, Acc@5: {best_test_acc5:.4f}")
    print("Training complete.")



