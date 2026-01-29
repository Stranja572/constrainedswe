import typing as T
from types import SimpleNamespace

import os
import pickle as pk
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.random import choice
from sklearn.model_selection import KFold, train_test_split
# from tdc.benchmark_group import dti_dg_group
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from featurizer import Featurizer
from featurizer.protein import FOLDSEEK_MISSING_IDX
from utils import get_logger

from Bio import SeqIO
import requests as r
from Bio import SeqIO
from io import StringIO

logg = get_logger()

def retrieve_protein_seq(ID):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+ID+".fasta"
    response = r.post(currentUrl)
    cData=''.join(response.text)

    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq,'fasta'))

    return str(pSeq[0].seq)


def get_task_dir(task_name: str, database_root: Path):
    """
    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    database_root = Path(database_root).resolve()

    task_paths = {
        "scl": database_root / "SCL",
    }

    return Path(task_paths[task_name.lower()]).resolve()


def target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # d_emb = [a[0] for a in args]
    t_emb = [a[0] for a in args]
    labs = [a[1] for a in args]

    # drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
    labels = torch.stack(labs, 0)

    return targets, labels



class SCLDataset(Dataset):
    def __init__(
        self,
        targets,
        labels,
        target_featurizer: Featurizer,
    ):
        self.targets = targets
        self.labels = labels

        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i])

        return target, label

class SCLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        # drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._all_path = Path("balanced.csv")

        self._target_column = "sequence"
        self._label_column = "target"
        self._split_column = "set"
        self._val_column = "validation"

        self.target_featurizer = target_featurizer

    def prepare_data(self):
        print("self.target_featurizer.path", self.target_featurizer.path)
        if self.target_featurizer.path.exists():
            logg.warning("Target featurizer already exists")
            return

        print("self._data_dir", self._data_dir)

        df_all = pd.read_csv(self._data_dir / self._all_path, **self._csv_kwargs)

        df_train = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] != True)]
        df_val = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] == True)]
        df_test  = df_all.loc[(df_all[self._split_column] == "test")]

        dataframes = [df_train, df_val, df_test]
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()

        if self._device.type == "cuda":
            self.target_featurizer.cuda(self._device)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):

        df_all = pd.read_csv(self._data_dir / self._all_path, **self._csv_kwargs)

        self.df_train = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] != True)]
        self.df_val = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] == True)]
        self.df_test  = df_all.loc[(df_all[self._split_column] == "test")]

        self._dataframes = [self.df_train, self.df_val, self.df_test]
        
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.target_featurizer.cuda(self._device)

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        unique_labels = self.df_train[self._label_column].unique()
        self.label_mapper = {x: i for i, x in enumerate(unique_labels)}

        if stage == "fit" or stage is None:
            self.data_train = SCLDataset(
                self.df_train[self._target_column],
                self.df_train[self._label_column].map(self.label_mapper),
                self.target_featurizer,
            )

            self.data_val = SCLDataset(
                self.df_val[self._target_column],
                self.df_val[self._label_column].map(self.label_mapper),
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = SCLDataset(
                self.df_test[self._target_column],
                self.df_test[self._label_column].map(self.label_mapper),
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

