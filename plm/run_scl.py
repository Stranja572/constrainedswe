import typing as T

import copy
import json
import os
import itertools
from argparse import ArgumentParser
import distutils.util
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torchmetrics
import wandb
from omegaconf import OmegaConf
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm.auto import tqdm

from data import (
    SCLDataModule,
    get_task_dir,
)
from featurizer import get_featurizer
from model import architectures as model_types
from utils import config_logger, get_logger, set_random_seed

logg = get_logger()

def add_args():

    parser = ArgumentParser('Subcellular Localization with ESM-2 and SWE aggregtion')

    parser.add_argument("--run-id", required=False, help="Experiment ID", dest="run_id", default="SCL")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file",
        default="configs/default_config.yaml",
    )

    # Logging and Paths
    log_group = parser.add_argument_group("Logging and Paths")

    log_group.add_argument(
        "--wandb-proj",
        help="Weights and Biases Project",
        dest="wandb_proj",
    )
    log_group.add_argument(
        "--wandb_save",
        help="Log to Weights and Biases",
        dest="wandb_save",
        type=lambda x:bool(distutils.util.strtobool(x)),
    )
    log_group.add_argument(
        "--log-file",
        help="Log file",
        dest="log_file",
    )
    log_group.add_argument(
        "--model-save-dir",
        help="Model save directory",
        dest="model_save_dir",
    )
    log_group.add_argument(
        "--data-cache-dir",
        help="Data cache directory",
        dest="data_cache_dir",
    )

    # Miscellaneous
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--r", "--replicate", type=int, help="Replicate", dest="replicate"
    )
    misc_group.add_argument(
        "--d", "--device", type=int, help="CUDA device", dest="device"
    )
    misc_group.add_argument(
        "--verbosity", type=int, help="Level at which to log", dest="verbosity"
    )
    misc_group.add_argument(
        "--checkpoint", default=None, help="Model weights to start from"
    )

    # Task and Dataset
    task_group = parser.add_argument_group("Task and Dataset")

    task_group.add_argument(
        "--task",
        choices=[
            "scl",
        ],
        type=str,
        help="Task name: scl",
    )

    # Model and Featurizers
    model_group = parser.add_argument_group("Model and Featurizers")

    model_group.add_argument(
        "--target-featurizer", help="Target featurizer", dest="target_featurizer"
    )
    model_group.add_argument(
        "--target-model-type", help="Target featurizer model type (for ESM featurizer only)", dest="target_model_type"
    )
    model_group.add_argument(
        "--model-architecture", help="Model architecture", dest="model_architecture"
    )
    model_group.add_argument(
        "--pooling", type=str, help="Pooling method", dest="pooling"
    )
    model_group.add_argument(
        "--num-ref-points", type=int, help="Size of the reference set", dest="num_ref_points"
    )
    model_group.add_argument(
        "--num-slices", type=int, help="Number of SWE slices", dest="num_slices"
    )
    model_group.add_argument(
        "--alpha-slack", type=float, help="slack alpha", dest="alpha_slack"
    )
    model_group.add_argument(
        "--dual-lr", type=float, help="Dual LR", dest="dual_lr"
    )
    model_group.add_argument(
        "--eps", type=float, help="eps", dest="eps"
    )
    model_group.add_argument(
        "--tau-softsort", type=float, help="tau_softsort", dest="tau_softsort"
    )

    # Training
    train_group = parser.add_argument_group("Training")

    train_group.add_argument("--epochs", type=int, help="number of total epochs to run")
    train_group.add_argument("-b", "--batch-size", type=int, help="batch size")
    train_group.add_argument("--shuffle", type=bool, help="shuffle data")
    train_group.add_argument("--num-workers", type=int, help="number of workers")
    train_group.add_argument("--every-n-val", type=int, help="validate every n epochs")

    train_group.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        help="initial learning rate",
        dest="lr",
    )
    train_group.add_argument(
        "--lr-t0", type=int, help="number of epochs to reset learning rate"
    )

    args = parser.parse_args()

    return args

def test(model, data_generator, metrics, device=None, classify=True):
    if device is None:
        device = torch.device("cpu")

    metric_dict = {}

    for k, met_class in metrics.items():
        met_instance = met_class(task="multiclass", num_classes=model.num_classes)
        met_instance.to(device)
        met_instance.reset()
        metric_dict[k] = met_instance

    model.eval()

    per_slice_distances = []
    for batch_ind, batch in tqdm(enumerate(data_generator), total=len(data_generator)):
        pred, label, _, _ = step(model, batch, device)


        if classify:
            label = label.int()
        else:
            label = label.float()

        for _, met_instance in metric_dict.items():
            met_instance(pred, label)

    results = {}
    for k, met_instance in metric_dict.items():
        res = met_instance.compute()
        results[k] = res
        # print(f"Metric {k}: {res}")

    for met_instance in metric_dict.values():
        met_instance.to("cpu")

    return results

def step(model, batch, device=None):
    if device is None:
        device = torch.device("cpu")

    target, label = batch
    pred, per_slice_distances, lambdas = model(target.to(device))
    label = Variable(torch.from_numpy(np.array(label))).to(device)

    return pred, label, per_slice_distances, lambdas

def wandb_log(m, do_wandb=True):
    if do_wandb:
        wandb.log(m)

def main():

    args = add_args()

    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)



    if config.pooling == 'swe':
        config.run_id = "{0}_swe_{1}_M{2}_L{3}_eps{4}_dualLR_{5}_alphalapsum{6}_tau_aggregation{7}".format(config.task, config.target_model_type, config.num_ref_points, config.num_slices, 
                                                                                                config.eps, config.dual_lr, config.alpha_lapsum, config.tau_aggregation)
    elif config.pooling == 'avg':
        config.run_id = "{0}_avg_{1}".format(config.task, config.target_model_type)

    elif config.pooling == 'fswe':
        config.run_id = "{0}_fswe_{1}_M{2}_L{3}".format(config.task, config.target_model_type, config.num_ref_points, config.num_slices)

    else:
        raise Exception
                                                                                            

    print("config.run_id + '_{}'.format(config.replicate)", config.run_id + '_{}'.format(config.replicate))                                                                                        

    # check if run is already done
    api = wandb.Api()

    try:
        runs = api.runs(path='nnlab/{}'.format(config.wandb_proj))
        _ = len(runs)
    except (ValueError, Exception) as e:
        # Project does not exist or other error; skip
        print(f"Project not found or error occurred: {e}")
        runs = []

    for run in runs:
        if run.name == config.run_id + '_{}'.format(config.replicate) and run.state in ["finished", "running"]:
            print('Run already completed or active! Terminating ...')
            raise Exception
        else:
            pass

    save_dir = f'{config.get("model_save_dir", ".")}/{config.run_id}'
    os.makedirs(save_dir, exist_ok=True)

    # Logging
    if "log_file" not in config:
        config.log_file = None
    else:
        os.makedirs(Path(config.log_file).parent, exist_ok=True)
    config_logger(
        config.log_file,
        "%(asctime)s [%(levelname)s] %(message)s",
        config.verbosity,
        use_stdout=True,
    )

    # Set CUDA device
    device_no = config.device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{device_no}" if use_cuda else "cpu")
    logg.info(f"Using CUDA device {device}")

    # Set random seed
    logg.debug(f"Setting random seed {config.replicate}")
    set_random_seed(config.replicate)

    # Load DataModule
    logg.info("Preparing DataModule")
    task_dir = get_task_dir(config.task, database_root=config.data_cache_dir)

    target_featurizer = get_featurizer(config.target_featurizer, save_dir=task_dir, per_tok=True, model_type=config.target_model_type)

    
    if config.task == "scl":
        config.classify = True
        config.watch_metric = "val/acc"
        config.latent_activation = "ReLU"
        datamodule = SCLDataModule(
            task_dir,
            target_featurizer,
            device=device,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
    else:
        raise Exception

    datamodule.prepare_data()
    datamodule.setup()

    # Load DataLoaders
    logg.info("Getting DataLoaders")
    training_generator = datamodule.train_dataloader()
    validation_generator = datamodule.val_dataloader()
    testing_generator = datamodule.test_dataloader()

    config.target_shape = target_featurizer.shape

    unconstrained_swgg_dict = {
        ('esm2_t33_650M_UR50D', 100, 32): 36.88346354166606,
        ('esm2_t33_650M_UR50D', 100, 64): 36.87337239583394,
        ('esm2_t33_650M_UR50D', 100, 128): 36.86653645833394,
    }

    if config.eps > 1000:
        effective_eps = config.eps
    else: # it's going to be a fraction of the unconstrained SWGG
        effective_eps = config.eps * unconstrained_swgg_dict[config.target_model_type, config.num_ref_points, config.num_slices]

    print("effective_eps", effective_eps)

    # Model
    logg.info("Initializing model")
    model = getattr(model_types, config.model_architecture)(
        config.target_shape,
        num_classes=config.num_classes,
        classify=config.classify,
        pooling=config.pooling,
        num_ref_points=config.num_ref_points,
        num_slices=config.num_slices,
        dual_lr=config.dual_lr,
        eps=effective_eps,
        alpha_lapsum=config.alpha_lapsum,
        tau_aggregation=config.tau_aggregation,
    )

    num_slices = config.num_slices

    
    if "checkpoint" in config:
        state_dict = torch.load(config.checkpoint)
        model.load_state_dict(state_dict)

    model = model.to(device)
    logg.info(model)

    # Optimizers
    logg.info("Initializing optimizers")
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=config.lr_t0
    )


    # Metrics
    logg.info("Initializing metrics")
    max_metric = 0
    model_max = getattr(model_types, config.model_architecture)(
        config.target_shape,
        num_classes=config.num_classes,
        classify=config.classify,
        pooling=config.pooling,
        num_ref_points=config.num_ref_points,
        num_slices=config.num_slices,
        dual_lr=config.dual_lr,
        eps=effective_eps,
        alpha_lapsum=config.alpha_lapsum,
        tau_aggregation=config.tau_aggregation,
    )
    model_max.load_state_dict(model.state_dict())

    if config.task == "scl":
        loss_fct = torch.nn.CrossEntropyLoss()
        val_metrics = {
            "val/acc": torchmetrics.Accuracy,
        }

        test_metrics = {
            "test/acc": torchmetrics.Accuracy,
        }
    else:
        raise Exception



    # Initialize wandb
    do_wandb = config.wandb_save and ("wandb_proj" in config)
    if do_wandb:

        logg.info(f"Initializing wandb project {config.wandb_proj}")
        wandb.init(
            project=config.wandb_proj,
            name=config.run_id + '_{}'.format(config.replicate),
            config=dict(config),
        )
        wandb.watch(model, log_freq=100)

    wandb.define_metric(config.watch_metric, summary="max") # save the *maximum* value of the watch metric in the summary dictionary (instead of last value)

    logg.info("Config:")
    logg.info(json.dumps(dict(config), indent=4))

    logg.info("Beginning Training")

    torch.backends.cudnn.benchmark = True

    # Begin Training
    start_time = time()
    for epo in range(config.epochs):
        model.train()
        epoch_time_start = time()

        # Main Step
        for i, batch in tqdm(
            enumerate(training_generator), total=len(training_generator)
        ):

            pred, label, per_slice_distances, lambdas = step(model, batch, device)

            loss = loss_fct(pred, label)

            constraint_violations = per_slice_distances - effective_eps

            lagrangian = loss + torch.sum(lambdas * constraint_violations)

            wandb_log(
                {
                    "train/step": (epo * len(training_generator) * config.batch_size)
                    + (i * config.batch_size),
                    "train/loss": loss,
                },
                do_wandb,
            )

            opt.zero_grad()
            lagrangian.backward()
            opt.step()


            wandb_log(
                {
                    "train/constraint_violations": constraint_violations.mean().item(),
                    "lambda_mean": lambdas.mean().item(),
                    "lambda_std": lambdas.std().item(),
                },
                do_wandb,
            )

            for l in range(num_slices):
                wandb_log(
                    {
                        "lambda_{}".format(l): lambdas[l].item(),
                    },
                    do_wandb,
                )

        lr_scheduler.step()
        config.dual_lr *= .95

        wandb_log(
            {
                "epoch": epo,
                "train/lr": lr_scheduler.get_lr()[0],
            },
            do_wandb,
        )
        logg.info(
            f"Training at Epoch {epo + 1} with loss {loss.cpu().detach().numpy():8f}"
        )
        logg.info(f"Updating learning rate to {lr_scheduler.get_lr()[0]:8f}")

        epoch_time_end = time()

        # Validation
        if epo % config.every_n_val == 0:
            with torch.set_grad_enabled(False):
                val_results = test(
                    model,
                    validation_generator,
                    val_metrics,
                    device,
                    config.classify,
                )

                val_results["epoch"] = epo
                val_results["Charts/epoch_time"] = (
                    epoch_time_end - epoch_time_start
                ) / config.every_n_val

                wandb_log(val_results, do_wandb)


                if val_results[config.watch_metric] > max_metric:
                    logg.debug(
                        f"Validation performance {val_results[config.watch_metric]:8f} > previous max {max_metric:8f}"
                    )
                    model_max.load_state_dict(model.state_dict())
                    max_metric = val_results[config.watch_metric]
                    model_save_path = Path(
                        f"{save_dir}/{config.run_id}_{config.replicate}_best_model.pt"
                    )
                    torch.save(
                        model_max.state_dict(),
                        model_save_path,
                    )
                    logg.info(f"Saving checkpoint model to {model_save_path}")

                logg.info(f"Validation at Epoch {epo + 1}")
                for k, v in val_results.items():
                    if not k.startswith("_"):
                        logg.info(f"{k}: {v}")

    end_time = time()

    # Testing
    logg.info("Beginning testing")
    try:
        with torch.set_grad_enabled(False):
            model_max = model_max.eval().to(device)

            test_start_time = time()
            test_results = test(
                model_max,
                testing_generator,
                test_metrics,
                device,
                config.classify,
            )
            test_end_time = time()

            test_results["epoch"] = epo + 1
            test_results["test/eval_time"] = test_end_time - test_start_time
            test_results["Charts/wall_clock_time"] = end_time - start_time
            wandb_log(test_results, do_wandb)

            logg.info("Final Testing")
            for k, v in test_results.items():
                if not k.startswith("_"):
                    logg.info(f"{k}: {v}")

    except Exception as e:
        logg.error(f"Testing failed with exception {e}")

    return model_max

if __name__ == "__main__":
    best_model = main()