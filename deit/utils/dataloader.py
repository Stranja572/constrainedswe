import os
from PIL import ImageOps
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_dataloaders(data_dir: str,  batch_size: int, num_workers: int, val_split: float = 0.1, seed: int = 0):
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
        # Secondary
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

    # Validation / test transform
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Load and split the training folder
    train_dir = os.path.join(data_dir, 'train')
    full_train = ImageFolder(train_dir, transform=train_transform)
    total = len(full_train)
    n_val = int(val_split * total)
    n_train = total - n_val

    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # override transforms for the validation subset
    val_ds.dataset.transform = val_transform

    # --- Load held-out “val” as test
    test_dir = os.path.join(data_dir, 'val')
    test_ds = ImageFolder(test_dir, transform=val_transform)

    # --- Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader
