import os
import numpy as np
import bisect
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from tqdm import tqdm
from typing import Callable, Union
import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

import quickdraw as QD


class QuickDrawDataSet(torch.utils.data.Dataset):
    def __init__(self, name, classes: list, max_drawings: int, transform: Callable, recognized=True):
        self.id = classes.index(name)
        self.datagroup = QD.QuickDrawDataGroup(name,
                                               max_drawings=max_drawings,
                                               recognized=recognized)
        self.max_drawings = max_drawings
        self.transform = transform

    def __len__(self):
        return self.datagroup.drawing_count

    def _get_single_item(self, index: int):
        img = self.datagroup.get_drawing(index).image
        return (self.transform(img), self.id)

    def __getitem__(self, index: Union[int, slice, np.ndarray]):
        if type(index) == slice or type(index) == np.ndarray:
            if type(index) == slice:
                index = range(index.start, index.stop, index.step or 1)
            return [self._get_single_item(i) for i in index]
        return self._get_single_item(index)


class QuickDrawDataAllSet(torch.utils.data.Dataset):
    def __init__(self, classes, max_drawings, transform, recognized=True):
        params = dict(
            max_drawings=max_drawings, transform=transform, classes=classes, recognized=recognized
        )
        self.groups = [QuickDrawDataSet(cls, **params) for cls in classes]
        self.offset = [0]
        self.count = 0
        for g in self.groups:
            self.count += len(g)
            self.offset.append(self.count)
        self.classes = classes

    def __len__(self):
        return self.count

    def get_labels(self):
        return self.classes

    def get_label(self, index: int):
        return self.classes[index]

    def get_single_item(self, index: int):
        gi = bisect.bisect_right(self.offset, index) - 1
        return self.groups[gi][index - self.offset[gi]]

    def __getitem__(self, index: Union[int, slice, np.ndarray]):
        if type(index) == slice or type(index) == np.ndarray:
            if type(index) == slice:
                index = range(index.start, index.stop, index.step or 1)
            return [self.get_single_item(i) for i in index]
        return self.get_single_item(index)


def split_index(indices, train_size: float) -> tuple:
    n = len(indices)
    i = int(n * train_size)
    return (indices[:i], indices[i:])


class DataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size: int, train_split_size: float = 0.8) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        index_all = np.arange(len(dataset))
        train_valid_idx, test_idx = split_index(
            index_all, train_size=train_split_size)
        train_idx, valid_idx = split_index(
            train_valid_idx, train_size=train_split_size)

        self.train_subsampler = SubsetRandomSampler(train_idx)
        self.valid_subsampler = SequentialSampler(valid_idx)
        self.test_subsampler = SequentialSampler(test_idx)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_subsampler)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.valid_subsampler)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.test_subsampler)
