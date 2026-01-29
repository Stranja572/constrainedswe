import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class DomainNetClipart(Dataset):
    """
    Reads one of your split‑txt files and loads images from deit/clipart/<class>/*.jpg
    """
    def __init__(self, root: str, list_file: str, transform=None):
        """
        root      : path to your deit/ folder
        list_file : e.g. "deit/clipart_train_split.txt"
        transform : torchvision transforms
        """
        self.transform = transform
        # each line: "clipart/<class>/<img>.jpg <label>"
        samples = [l.strip().split() for l in Path(list_file).read_text().splitlines() if l]
        self.samples = [
            (os.path.join(root, rel), int(lbl))
            for rel, lbl in samples #each line in the train/val file has number next to it corresponding to label
        ]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB") #allows raw image pixel to properly be converted to 3 number tensor (RGB)
        if self.transform:
            img = self.transform(img)
        return img, label