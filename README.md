# Constrained Sliced Wasserstein Embedding

This repository contains the experiments and implementation code for **Constrained Sliced Wasserstein Embedding**.

## Abstract

> Sliced Wasserstein (SW) distances offer an efficient method for comparing high-dimensional probability measures by projecting them onto multiple 1-dimensional probability distributions. However, identifying informative slicing directions has proven challenging, often requiring a large number of slices to achieve desirable performance, thereby increasing computational complexity. We introduce a constrained learning approach to optimize the slicing directions for SW distances. Specifically, we constrain the slice-wise transport costs corresponding to the lifted 1-dimensional transport plans in the original space, ensuring meaningful slicing directions. By leveraging continuous relaxations of these transport plans, we enable a gradient-based primal-dual approach to train the slicer parameters, alongside the remaining model parameters. We demonstrate how this constrained slicing approach can be applied to pool high-dimensional embeddings into fixed-length, permutation-invariant representations. Numerical results on foundation models trained on point clouds, images, and protein sequences showcase the efficacy of the proposed constrained learning approach in learning more informative slicing directions. 

## Create/Activate Virtual Environment
```bash
conda env create -f environment.yml
conda activate constrained-swe
```
   
## Running the Experiments

All commands below assume you have already created and activated the conda environment (see above).

### Point Cloud Classification on ModelNet40 with Point Cloud Transformers (PCTs)

1) Download + preprocess ModelNet40 (from the `modelnet` directory):

```bash
cd modelnet

python download_ModelNet40.py
python ModelNet40_data.py
```
2) Train (run from `modelnet` directory)
```bash
python -u train.py \
  --seed <seed> \
  --pooling <pooling_method> \
  --projections <num_projections> \
  --tau_aggregation 0.1 \
  --batch_size 64 \
  --epochs 200 \
  --eps_fraction <eps_fraction>
```

<pooling_method> options: `SWE`, `constrained_swe`, `FSWE`, `GAP`

`--eps_fraction` expects a floating-point value (decimal), e.g. `0.375` (`6/7`).



### Image Classification on DomainNet Clipart with Vision Transformers (DeiT/ViTs)

This setup uses file lists (`clipart_*_split.txt`) to define train/val splits.

1) Download DomainNet Clipart:

```bash
cd <repo_root>

mkdir -p deit/data
cd deit/data

wget http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip
unzip clipart.zip
```

2) Create a reproducible 90/10 train/val split
```bash
cd <repo_root>
python deit/make_val_split.py
```
Create a reproducible 90/10 train/val split:
Next, extract 10% of the training set as validation by running `make_val_split.py` to get the corresponding train and validation data. Then, run `organize_val.py` to separate the picture into their respective classes, which is necessary for the code to run.

Run the experiment with the following command from the `deit` directory:

3) Train (run from the `deit` directory)

Pooling method options: `SWE`, `constrained_SWE`, `GAP`, `FSWE`

```bash
cd <repo_root>/deit

python -u main.py \
  --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" \
  --pooling <pooling_method> \
  --projections <num_projections> \
  --tau_aggregation 0.01 \
  --batch_size 512 \
  --epochs 80 \
  --eps_fraction <eps_fraction> \
  --unconstrained_violation 27.9783 \
  --num_workers <num_workers>
```
<pooling_method> options: `SWE`, `constrained_swe`, `FSWE`, `GAP`, `cls`

`--eps_fraction` expects a floating-point value (decimal), e.g. `0.375` (`6/7`).

### Subcellular Localization with Protein Language Models (PLMs) 

The `download_data.py` script can be used to download the dataset for the experiments into a new folder called `datasets` by running

```bash
python download_data.py --to datasets --benchmarks scl
```

The hyperparameters, such as the pooling method and number of reference points, can be adjusted via command-line parameters and the configuration files under the `config` directory. To run the experiment, use the following command from the `plm` directory: 

```bash
python run_scl.py --config config/scl_esm2.yaml --pooling swe --num-ref-points 100 --target-model-type esm2_t33_650M_UR50D
```

**NOTE:** The first training epoch might take longer than usual because it creates all the required interpolation matrices for SWE corresponding to different sequence lengths in the dataset. Once that is done, the following epochs will be much faster.
    


   