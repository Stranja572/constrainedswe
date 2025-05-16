# Constrained Sliced Wasserstein Embedding

This repository contains the experiments and implementation code for **Constrained Sliced Wasserstein Embedding**.

## Abstract

> Sliced Wasserstein (SW) distances offer an efficient method for comparing high-dimensional probability measures by projecting them onto multiple 1-dimensional probability distributions. However, identifying informative slicing directions has proven challenging, often necessitating a large number of slices to achieve desirable performance and thereby increasing computational complexity. We introduce a constrained learning approach to optimize the slicing directions for SW distances. Specifically, we constrain the 1D transport plans to approximate the optimal plan in the original space, ensuring meaningful slicing directions. By leveraging continuous relaxations of these permutations, we enable a gradient-based primal-dual approach to train the slicer parameters, alongside the remaining model parameters. We demonstrate how this constrained slicing approach can be applied to pool high-dimensional embeddings into fixed-length permutation-invariant representations. Numerical results on foundation models trained on images, point clouds, and protein sequences showcase the efficacy of the proposed constrained learning approach in learning more informative slicing directions.

## Create/Activate Virtual Environment
```bash
conda env create -f environment.yml
conda activate constrained-swe
```
   
## Running the Experiments


### Image Classification with Vision Transformers (ViTs)

Download Tiny ImageNet with the following command:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

For compatability with our code, restructure the validation set (which is used as the test set) by placing images into subdirectories representing their respective classes. 

Run the experiment with the following command from the `deit` directory:

```bash
python -u main.py \
--data_dir <dataset_directory_path>
--classification <classification> \
--batch_size <batch_size> \
--epochs <epochs> \
--primal_lr <primal_learning_rate> \
--slack_lr <slack_learning_rate> \
--dual_lr <dual_learning_rate> \
--alpha <alpha> \
--epsilon <epsilon> \
--num_projections <num_projections> \
--tau_softsort <tau_softsort> \
--layer_stop <layer_stop> \
--seed <seed>
```

Additional command arguments can be found in `main.py`.

### Point Cloud Classification with Point Cloud Transformers (PCTs)

Use the `download_ModelNet40.py.` and `ModelNet40_data.py` scripts to download and preprocess the data. 

Then, run the following command from the `modelnet` directory:

```bash
python train.py
```


### Subcellular Localization with Protein Language Models (PLMs) 

The `download_data.py` script can be used to download the dataset for the experiments into a new folder called `datasets` by running

```bash
python download_data.py --to datasets --benchmarks scl
```

The hyperparameters, such as the number of points in the reference set and the pre-trained PLM backbone, can be adjusted via command-line parameters, as well as the configuration files under `config`. To run the experiment, use the following command from the `plm` directory: 

```bash
python run_scl.py --config config/scl_esm2.yaml --pooling swe --num-ref-points 100 --target-model-type esm2_t6_8M_UR50D
```

**NOTE:** The first training epoch might take longer than usual because it creates all the required interpolation matrices for SWE corresponding to different seequence lengths in the dataset. Once that is done, the following epochs will be much faster.
    


   