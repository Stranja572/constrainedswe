import argparse
import time
import wandb
import torch
from train import train_and_evaluate
from utils.dataloader import get_dataloaders
from model import ConstrainedDeiT

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default = "None")
    parser.add_argument("--num_classes", type=int, default=200) #Imagenet has 200 classes
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--classification", type=str, default="constrained_swe")
    parser.add_argument("--epochs", type=int, default=90) #in each epoch we update params, lambdas, and slacks
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--primal_lr", type=float, default=1e-3, help="ηΘ: learning rate for pooling & classifier")
    parser.add_argument("--slack_lr", type=float, default=1e-3, help="ηs: learning rate for slack variables")
    parser.add_argument("--dual_lr", type=float, default=0.01, help="ηλ: dual variable step size")
    parser.add_argument("--alpha", type=float, default=0.1, help="Slack regularization coefficient α")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Constraint tolerance vector ϵ")
    parser.add_argument("--num_ref_points", type=int, default=196, help="Number of reference points in SWE") #196 ref points -> no interpolation
    parser.add_argument("--num_projections", type=int, default=8, help="Number of projections in SWE")
    parser.add_argument("--tau_softsort", type=float, default=1e-2)
    parser.add_argument("--wandb", type=lambda x: x.lower() in ['true','1','yes'], default=False, help="Enable Weights & Biases logging (True or False)")
    parser.add_argument("--wandb_key", type=str, default="None", help="W&B API key") #must add wandb key
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset split and training")
    parser.add_argument("--val_split", type=float, default=0.1,  help="Fraction of train set to use for validation split")
    parser.add_argument("--output_dir",    type=str, default="./checkpoints_real")
    parser.add_argument("--layer_stop",    type=int, default=11)
    parser.add_argument("--embedding",    type=str, default="flatten")
    parser.add_argument("--parallel", type=lambda x: x.lower() in ['true','1','yes'], default=True)

    return parser.parse_args()

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.classification == "swe" or args.classification == "cls":
        args.epsilon = 10000
    
    if args.classification != "cls":
        fname = f"deit_tiny_{args.classification}_seed{args.seed}_epsilon_{args.epsilon}_{args.batch_size}_alpha{args.alpha}__duallr{args.dual_lr}_tau{args.tau_softsort}_{args.num_projections}slices"
    else:
        fname = f"deit_tiny_{args.classification}_seed{args.seed}_bs{args.batch_size}"

    print(fname)
    
    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")


    if args.wandb:
        assert args.wandb_key, "--wandb_key is required when --wandb is set"
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="imagenet-tiny-real",
            entity="constrained-swe",
            name=fname,
            config=vars(args)
        )
        time.sleep(5)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    model = ConstrainedDeiT(
        num_classes=args.num_classes,
        num_ref_points=args.num_ref_points,
        num_projections=args.num_projections,
        tau_softsort=args.tau_softsort,
        classification = args.classification,
        layer_stop = args.layer_stop,
        embedding = args.embedding,
        parallel = args.parallel
    )

    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs — using DataParallel")
        model = torch.nn.DataParallel(model)

    train_and_evaluate(args, model, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()