
# ## Importings
import torch.nn.functional as F
from typing import Tuple
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
import random
import bisect
from typing import Union
import argparse
import quickdraw as QD
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, LearningRateFinder, Callback
)
from lightning.pytorch.loggers import WandbLogger
import warnings
from lightning.pytorch.tuner import Tuner
from PIL import Image as PILImage
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import wandb
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Optional, Callable, List, Tuple

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms as T
from tqdm import tqdm
import lightning as L

from glob import glob
import gc
gc.enable()

print("pytorch version:", torch.__version__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Traning options",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--dataset-samples", action="store", type=int,
                    help="the number of samples for data set per class", default=2000)
parser.add_argument("-c", "--classes", action="store", type=str,
                    help="the list of classes to train")
parser.add_argument("-e", "--epochs", action="store", type=int, default=50)
parser.add_argument("-b", "--batch-size", action="store", type=int, default=64)
parser.add_argument("-l", "--lr", "--learning-rate", action="store", type=float,
                    help="learning rate", default=1e-3)
parser.add_argument("-d", "--dropout", action="store", type=float,
                    help="drop out", default=0.2)
parser.add_argument("-m", "--model-name", action="store", help="model name")
parser.add_argument("--model-state", action="store",
                    help="model state location to continue")
parser.add_argument("--checkpoint", action="store",
                    help="[lightning] model checkpoint")
parser.add_argument("--logs", action="store",
                    help="path to save logs", default="log.json")
parser.add_argument("--save-figures", action="store",
                    help="directory name to save figures", default="figures")
parser.add_argument("--wandb-api", action="store",
                    help="API token to use wandb, or use environment variable WANDB_API_KEY")
parser.add_argument("--batch-norm", action="store_true",
                    help="use batch normalization")
parser.add_argument("--progress-bar", action="store_true",
                    help="Show progress bar")
parser.add_argument("--seed", type=int, default=1234,
                    metavar="S", help="random seed (default: 1234)")

config = parser.parse_args()

with open(config.classes, 'r') as f:
    CLASSES = sorted(filter(lambda s: bool(len(s)), f.readlines()))
    CLASSES = list(map(lambda s: s.replace('\n', ''), CLASSES))
    print('predict classes:', CLASSES)


class QuickDrawDataSet(torch.utils.data.Dataset):
    def __init__(self, name, max_drawings, transform, recognized=True, classes=CLASSES):
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


# Horizontal line as divider
HR = "\n" + ("-" * 30) + "\n"

# Options
SEED = int(config.seed)

# - Dataset
OUT_CLASSES = len(CLASSES)

# - Training
EPOCH_RUNS = int(config.epochs)
BATCH_SIZE = int(config.batch_size)
LEARNING_RATE = float(config.lr)
USE_BATCH_NORM = bool(config.batch_norm)

# - Load model
MODEL_CHECKPOINT = config.checkpoint
MODEL_OUTPUT_NAME = config.model_name or f'model'

# - Logging
LOG_JSON_PATH = config.logs
FIG_OUTPUT_DIR = config.save_figures
PLOT_PATH = os.path.join(FIG_OUTPUT_DIR, "plot.png")
LOG_SAVE_INTERVAL: int = 1
PLOT_SAVE_INTERVAL: int = 2
MODEL_SAVE_INTERVAL: int = 5
SHOW_PROGRESS: bool = config.progress_bar

WB_API: str = config.wandb_api or os.environ['WANDB_API_KEY']

# Parameters
BATCH_SIZE = 64

# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


torch.manual_seed(SEED)

wandb.login(key=WB_API)

# Logger
wandb_logger = WandbLogger(
    # set the wandb project where this run will be logged
    project="Quick Draw CNN",
    tags=["CNN", "Adam", "Tuner", "Normalized", "32x32"],
    # track hyperparameters and run metadata
    config=vars(config),
    log_model='all',
)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    return "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


device = get_device_name()
print('device:', device, HR)

encode_image = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    #   T.Normalize(mean=[0.485, 0.456, 0.406],
    #               std=[0.229, 0.224, 0.225]),
    T.Normalize(0.5, 0.5)
])
decode_image = T.Compose([
    T.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),
    T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
    T.ToPILImage()
])

# ### Load dataset
dataset_all = QuickDrawDataAllSet(CLASSES,
                                  max_drawings=config.dataset_samples,
                                  transform=encode_image)


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


def create_image_grid(
    fig,
    nrows_ncols: Tuple[int, int],
    x: torch.Tensor,
    y_true: list,
    y_probs: list,
) -> tuple:
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.4)
    y_pred = np.argmax(y_probs, axis=1)

    for ax, im, yt, yp in zip(grid, x, y_true, y_pred):
        img = decode_image(im)
        ax.imshow(img)
        if yt == yp:
            ax.set_title(dataset_all.get_label(yt))
        else:
            ax.set_title(
                f"P:{dataset_all.get_label(yp)}\nT: {dataset_all.get_label(yt)}")
        ax.axis('off')
    return grid


print(HR)

# Define CNN model


class CNNModel(pl.LightningModule):
    def __init__(self, output_classes: int, dropout=0.2):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # (3, 32, 32)
            nn.Conv2d(3, 16, 2, padding=1),  # (32, 33, 33)
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),  # (64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 16, 16)

            nn.Conv2d(32, 64, 2),  # (128, 15, 15)
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),  # (256, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (256, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'valid_loss': loss, 'valid_acc': acc})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        acc = self.accuracy_score(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_acc': acc})
        return loss

    def accuracy_score(self, probs, y):
        return torch.sum(torch.argmax(probs, dim=1) == y) / len(y)

    def configure_optimizers(self):
        lr = self.hparams.lr
        print("model.configure_optimizers.lr =", lr)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.85)
        return [optimizer], [scheduler]


# ## Load model
model = CNNModel(len(dataset_all.get_labels()), dropout=config.dropout)
model.to(device)
model.save_hyperparameters(config)
print("Model: ===============================")
print(model, HR)

wandb_logger.watch(model, log="all")
wandb_logger.log_graph(model=model)

lr_callback = LearningRateMonitor()
checkpoint_callback = ModelCheckpoint(monitor="valid_acc", mode="max")
earlystop_callback = EarlyStopping(monitor="valid_loss")


class AutoLearningRateFinder(LearningRateFinder):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


lr_find_callback = AutoLearningRateFinder(
    min_lr=1e-6, max_lr=0.1, attr_name="lr",
)


class SaveBatchImageCallback(Callback):
    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           dataset: L.LightningModule,
                           outputs: STEP_OUTPUT | None,
                           batch: tuple,
                           batch_idx: int,
                           dataloader_idx: int = 0) -> None:
        if batch_idx != 0:
            return
        x, y = batch
        y_probs = trainer.model(x)
        fig = plt.figure(figsize=(10, 10))
        fig.tight_layout(pad=0.1)
        axes = create_image_grid(fig,
                                 (5, 5),
                                 x,
                                 y.detach().cpu().numpy(),
                                 y_probs.detach().cpu().numpy())
        imgs = [wandb.Image(fig)]
        wandb_logger.log_image(key="batch_image", images=imgs)
        plt.close(fig)


# Trainer
trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCH_RUNS,
    # use_distributed_sampler=True,
    enable_checkpointing=True,
    enable_model_summary=True,
    enable_progress_bar=True,
    log_every_n_steps=5,
    logger=wandb_logger,
    callbacks=[
        lr_callback,
        checkpoint_callback,
        earlystop_callback,
        # lr_find_callback, # should be removed if use Tuner
        SaveBatchImageCallback(),
    ],
)
dm = DataModule(dataset=dataset_all, batch_size=BATCH_SIZE)

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, datamodule=dm)
print("lr_finder:", lr_finder.results)

fig = lr_finder.plot(suggest=True)
fig.savefig("lr_find.png")
plt.close(fig)

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.lr = new_lr
model.configure_optimizers()

trainer.fit(model, datamodule=dm, ckpt_path=MODEL_CHECKPOINT)
trainer.validate(model, dm, verbose=True)
trainer.test(model, datamodule=dm, verbose=True)

print("Train terminated.")
# wandb.alert(title='Train terminated', text='Check it out')
