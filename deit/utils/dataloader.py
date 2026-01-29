import os
import random
from pathlib import Path
from PIL import Image, ImageOps
import torch
from utils.clipart import DomainNetClipart
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_dataloaders(dataset: str,  batch_size: int, num_workers: int, val_split: float = 0.1, seed: int = 0):
    """
    - Splits data_dir/train into train + validation according to val_split.
    - Loads data_dir/val as the final test set.
    - All images are transformed to 224×224.
    """
    
    img_size = 224
    
    # GaussianBlur kernel ~0.1 * img_size, must be odd
    k = max(3, int(0.1 * img_size))
    if k % 2 == 0:
        k += 1

    # === Train transform: primary crop+flip, secondary one-of-three, jitter, tensor+normalize
    train_transform = transforms.Compose([
        # Primary
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # Secondary: exactly one of {Grayscale, Solarize, Blur}
        transforms.RandomChoice([
            transforms.RandomGrayscale(p=1.0),
            transforms.Lambda(lambda img: ImageOps.solarize(img, threshold=128)),
            transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0)),
        ]),
        # Photometric jitter ~80% of the time
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8
        ),
        # To tensor + normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # === Validation / test transform
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

   
    root = "deit" #edit based on your actual root
    # build each split from its txt list
    train_ds = DomainNetClipart(root, os.path.join(root, "clipart_train_split.txt"), train_transform)
    val_ds   = DomainNetClipart(root, os.path.join(root, "clipart_val_split.txt"),   val_transform)
    test_ds  = DomainNetClipart(root, os.path.join(root, "clipart_test.txt"),        val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=8)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader, test_loader