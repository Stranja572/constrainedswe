import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import os
import itertools
import random
import numpy as np
from modelnet.cswe import ConstrainedSWE
from pct import NaivePCT, NaivePCT_cls

from collections import defaultdict
from tqdm import tqdm
from ModelNet40_data import ModelNet40
from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)


import wandb


batch_size = 64
base_random_seed = 2054
num_epochs = 200
output_dim = 256
lr = 1e-3
num_classes = 40


class Backbone(nn.Module):
    def __init__(self, backbone_type, input_dim=3, output_dim=3):
        super(Backbone, self).__init__()

        if backbone_type == 'PCT':
            self.backbone = NaivePCT(d_out=output_dim)
        elif backbone_type == 'PCT_CLS':
            self.backbone = NaivePCT_cls(d_out=output_dim)
        # elif backbone_type == 'ST':
        #     self.backbone = SetTransformer(d_in=input_dim, d_out=output_dim)
        else:
            raise ValueError('Backbone type {} not implemented!'.format(backbone_type))

    def forward(self, x):
        return self.backbone(x)

class Pooling(nn.Module):
    def __init__(self, pooling, d_in=1, num_projections=1, num_ref_points=1, tau_softsort=1):

        super(Pooling, self).__init__()

        self.pooling = pooling
        # pooling mechanism
        if 'CSWE' in pooling:
            self.output_dim = num_ref_points * num_projections 

            # Dual and slack variables
            self.register_buffer('lambdas', torch.zeros(num_projections))#makes sure its not a parameter (no gd) 
            self.slacks = nn.Parameter(torch.zeros(num_projections))

            # Constrained SWE pooling
            self.pool = ConstrainedSWE(
                d_in=d_in,
                num_ref_points=num_ref_points,
                num_projections=num_projections,
                tau_softsort=tau_softsort
            )
            self.pool = ConstrainedSWE(d_in, num_ref_points, num_projections, tau_softsort)
            self.num_outputs = num_projections * num_ref_points
        elif 'GAP' in pooling or 'CLS' in pooling:
            self.num_outputs = d_in
        else:
            raise ValueError('Pooling type {} not implemented!'.format(pooling))

    def forward(self, P):
        """
        Input size: B x N x d_in
        B: batch size, N: # elements per set, d_in: # features per element

        Output size: B x self.num_outputs
        """
        if self.pooling == 'GAP':
            U = torch.mean(P, dim=1)
            violations = None
        elif self.pooling == 'CLS':
            U = P # Identity
            violations = None
        else:
            pooled, violations = self.pool(P)
            U = pooled.reshape(-1, self.num_outputs)

        return U, violations

def train_test(b, e, n, random_seed, num_points_per_set, alpha_slack, dual_lr, eps, tau_softsort):#, gpu_index):

    device = torch.device('cuda:0')

    print("device", device)

    project_name = 'constrained_swe_modelnet40_pretrained'

    # create results directory if it doesn't exist
    results_dir = './results/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}/'.format(project_name, b, e, n, random_seed, num_points_per_set, alpha_slack, dual_lr, eps, tau_softsort)
    os.makedirs(results_dir, exist_ok=True)


    # Set the random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    res_path = results_dir + 'results.json'#.format(random_seed)
    pooling_path = results_dir + 'pooling.pt'#.format(random_seed)

    if os.path.exists(res_path):
        try:
            epochMetrics = torch.load(res_path, weights_only=False)
            if len(epochMetrics['test', 'loss']) == num_epochs:
                print("Run already completed!")
                return
        except:
            pass    

    print("params", b, e, n, random_seed, num_points_per_set, alpha_slack, dual_lr, eps, tau_softsort)

    config_dict = {
                "num_points_per_set": num_points_per_set,
                "backbone": b, 
                "pooling": e, 
                "num_slices": n, 
                "alpha_slack": alpha_slack, 
                "dual_lr": dual_lr, 
                "eps": eps,
                "tau_softsort": tau_softsort,
                "random_seed": random_seed
            }

    run = wandb.init(
            project=project_name,
            # Track hyperparameters and run metadata
            config=config_dict,
            name = str(config_dict)
        )

    # get the datasets
    phases = ['train', 'valid', 'test']
    dataset = {}
    for phase in phases:
        dataset[phase] = ModelNet40(num_points_per_set, partition=phase)

    # create the dataloaders
    loader = {}
    for phase in phases:
        if phase == 'train':
            shuffle = True
        else:
            shuffle = False
        loader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=shuffle)

    # create the modules
    backbone = Backbone(b, output_dim=output_dim)
    if b == 'PCT_CLS':
        missing, unexpected = backbone.load_state_dict(torch.load('./ckpts/backbone.pt'), strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        backbone.load_state_dict(torch.load('./ckpts/backbone.pt'), strict=False)
    else:
        backbone.load_state_dict(torch.load('./ckpts/backbone.pt'))
    for name, param in backbone.named_parameters():
        print(name)
        if name != 'cls_token':
            param.requires_grad = False
    if b == 'PCT_CLS':
        backbone.backbone.cls_token.requires_grad = True
  
    pooling = Pooling(e, d_in=output_dim, num_projections=n, num_ref_points=num_points_per_set, tau_softsort=tau_softsort)
    classifier = nn.Linear(pooling.num_outputs, num_classes)


    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    # start training
    criterion = nn.CrossEntropyLoss()
    params = [backbone.backbone.cls_token] + \
             list(classifier.parameters()) 

    optim = Adam(params, lr=lr) #makes the slacks parameters, same as nn.Param pretty much
    scheduler = StepLR(optim, step_size=50, gamma=0.5)

    epochMetrics = defaultdict(list)

    best_val_acc = 0.0
    best = {}
    best['val'] = best_val_acc
    for epoch in tqdm(range(num_epochs)):
        for phase in ['train', 'valid', 'test']:

            if phase == 'train':
                # backbone.train()
                pooling.train()
                classifier.train()
            else:
                # backbone.eval()
                pooling.eval()
                classifier.eval()

            loss_ = []
            acc_ = []
            # constraint_violations_ = []

            for i, data in enumerate(loader[phase]):

                # print(i)

                # if phase == 'train' and i >=10:
                #     break

                # zero the parameter gradients
                optim.zero_grad()

                x, y = data

                x = x.to(device).to(torch.float)
                y = y.to(device).squeeze()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # pass the sets through the backbone and pooling
                    z = backbone(x)
                    v, per_slice_distances = pooling(z)
                    logits = classifier(v)
                    loss = criterion(logits, y)

                    # constraint_violations = per_slice_distances - (eps + slacks)

                    lagrangian = loss #+ torch.sum(lambdas * constraint_violations) + 0.5 * alpha_slack * torch.linalg.norm(slacks) ** 2

                    acc = (1. * (torch.argmax(logits, dim=1) == y)).mean().item()

                    # backpropogation only in training phase
                    if phase == 'train':
                        # Backward pass
                        lagrangian.backward()
                        # 1-step gradient descent
                        optim.step()
                    
                    # lambdas += dual_lr * constraint_violations.detach()
                    # lambdas.data.clamp_(min=0) 
                    # slacks.data.clamp_(min=0)

                # save losses and accuracies
                loss_.append(loss.item())
                acc_.append(acc)
                # constraint_violations_.append(constraint_violations.mean().item())

            
            epochMetrics[phase, 'loss'].append(np.mean(loss_))
            epochMetrics[phase, 'acc'].append(np.mean(acc_))
            # epochMetrics[phase, 'constraint_violations'].append(np.mean(constraint_violations_))
            # epochMetrics[phase, 'lambdas'].append(lambdas.detach().cpu().numpy())
            # epochMetrics[phase, 'slacks'].append(slacks.detach().cpu().numpy())
            

            wandb.log({"{} loss".format(phase): np.mean(loss_),
                       "{} acc".format(phase): np.mean(acc_),
                    #    "{} constraint_violations".format(phase): np.mean(constraint_violations_)
                       })

            # if phase == 'train':
                # wandb.log({"mean_lambda": np.mean(lambdas.detach().cpu().numpy()),
                #         "mean_slack": np.mean(slacks.detach().cpu().numpy())
                #         })
                # for l in range(n):
                #     wandb.log({"lambda_{}".format(l): lambdas[l].item(),
                #        "slack_{}".format(l): slacks[l].item()
                #        })
                


        scheduler.step()


        torch.save(epochMetrics, res_path)
        if epochMetrics['valid', 'acc'][-1] > best_val_acc:
            torch.save(pooling.state_dict(), pooling_path)
            best_val_acc = epochMetrics['valid', 'acc'][-1]
            best['val'] = best_val_acc
            best['test'] = epochMetrics['test', 'acc'][-1]

    torch.save(best, results_dir +'best.json')
    return epochMetrics


def main():

    backbones = ['PCT_CLS'] # ['PCT', 'ST']
    embeddings = ['CLS']
    num_projections_range = [64]# [4, 8, 16, 32, 64]
    random_seeds = [30000] # [0, 10000, 20000]
    num_points_per_set_range = [512] # points per set
    alpha_slack_values = [1] # [0.1, 1]
    dual_lr_values = [0.001] #[0.01, 0.001]
    # eps_values = [1, 3.5, 5, 7, 10000]
    eps_values = [7]
    tau_softsort_values = [1e-3] #[1e-2, 1e-3]

    params_all = list(itertools.product(backbones,
                                        embeddings,
                                        num_projections_range,
                                        random_seeds,
                                        num_points_per_set_range,
                                        alpha_slack_values,
                                        dual_lr_values,
                                        eps_values,
                                        tau_softsort_values
                                        ))



    # remove redundant parameters (esp with eps=1000, i.e., unconstrained)
    params = []
    encountered_unconstrained_random_seeds = []
    for p in params_all:
        if p[7] < np.max(eps_values):
            params.append(p)
        else:
            num_projections, random_seed = p[2], p[3]
            if (num_projections, random_seed) not in encountered_unconstrained_random_seeds:
                params.append(p)
                encountered_unconstrained_random_seeds.append((num_projections, random_seed))




    print(params)
    print('Number of parameter/random seed combinations:', len(params))


    num_processes = 1 #3
    print('Now starting the code using {} parallel processes...'.format(min(num_processes, len(params))))

    pool = Pool(num_processes)
    all_results = pool.starmap(train_test, params)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()