
# ## Importings
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

from sklearn.metrics import accuracy_score
from typing import Any, Optional, Callable

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import lightning as L

from sklearn.preprocessing import LabelEncoder
from models import LitLSTM

from models.SimpleLSTM import SimpleLSTM, SimpleLSTMBn
from trainer.callbacks import EarlyStopper
from trainer.dataloader import StrokesDataset
from utils import (
    draw_image_grid, draw_image_grid_with_probs, get_device_name, get_filename, makedirs, reconstruct_to_images, reconstruct_to_gif
)

from args import args as config

from glob import glob
import gc
gc.enable()

print("pytorch version:", torch.__version__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Horizontal line as divider
HR = "\n" + ("-" * 30) + "\n"


# Options
SEED = int(config.seed)

# - CUDA
CUDA_GPU_ID = int(config.cuda)

# - Dataset
DATASET_DIR = config.dataset
MAX_STROKES_LEN = int(config.n_strokes)
TRAIN_SAMPLES_PER_CLASS = int(config.train_samples)
VALID_SAMPLES_PER_CLASS = int(config.valid_samples)
OUT_CLASSES = int(config.classes)

# - Training
EPOCH_RUNS = int(config.epochs)
BATCH_SIZE = int(config.batch_size)
LEARNING_RATE = float(config.lr)
USE_BATCH_NORM = bool(config.batch_norm)

# - Load model
PREVIOUS_MODEL_STATE = config.model_state
MODEL_OUTPUT_NAME = config.model_name or f'model_{MAX_STROKES_LEN}'

# - Logging
LOG_JSON_PATH = config.logs
FIG_OUTPUT_DIR = config.save_figures
PLOT_PATH = os.path.join(FIG_OUTPUT_DIR, "plot.png")
LOG_SAVE_INTERVAL: int = 1
PLOT_SAVE_INTERVAL: int = 2
MODEL_SAVE_INTERVAL: int = 5
SHOW_PROGRESS: bool = config.progress_bar

WB_API: str = config.wandb_api or os.environ['WANDB_API_KEY']


# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


torch.manual_seed(SEED)


device = get_device_name()

if torch.cuda.is_available():
    device = f'cuda:{CUDA_GPU_ID}'
    torch.cuda.set_device(device)

print('device:', device, HR)

wandb.login(key=WB_API)

# ## Prepare to collect dataset


files = os.listdir(DATASET_DIR)
files = sorted(list(map(lambda p: os.path.join(DATASET_DIR, p), files)))


print("dataset shape of train/valid/test (sample)")
for filepath in files:
    npz = np.load(filepath, encoding='latin1', allow_pickle=True)
    print(filepath, npz["train"].shape, npz["valid"].shape, npz["test"].shape)
    break
print(HR)


def filename_to_label(path: str) -> str:
    return get_filename(path, ext=False)


# ### Label encoding


labels = list(map(filename_to_label, files))
labels[:10]


word_encoder = LabelEncoder()
word_encoder.fit(labels)
print('target words for output:', len(word_encoder.classes_), '=>',
      ', '.join([x for x in word_encoder.classes_]), HR)


# ### Load dataset (train/valid)


TRAIN_FILES = sorted(files)
print("Files used in dataset", f"({len(TRAIN_FILES)}):")
print('\n'.join(TRAIN_FILES[:3]))
print('...')
print('\n'.join(TRAIN_FILES[-3:]), HR)


class SketchesDataModule(L.LightningDataModule):
    def __init__(self, batch_size, encoder) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.encoder = encoder

    def setup(self, stage: str) -> None:
        self.train_dataset = self.load_dataset("train",
                                               max_row=TRAIN_SAMPLES_PER_CLASS)
        self.valid_dataset = self.load_dataset("valid",
                                               max_row=VALID_SAMPLES_PER_CLASS)
        self.test_dataset = self.load_dataset("test")
        return super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = SubsetRandomSampler(range(len(self.train_dataset)))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def load_dataset(self, is_for, max_row: int = 0):
        return StrokesDataset(
            TRAIN_FILES,
            is_for,
            max_row=max_row,
            strokes=MAX_STROKES_LEN,
            y_transform=self.encoder)

    def teardown(self, stage: str) -> None:
        del self.train_dataset
        del self.valid_dataset
        del self.test_dataset
        return super().teardown(stage)


ROWS = 6
COLS = 6


def save_result_image(
    x: list,
    y_true: list,
    y_pred: list,
    output_path: str,
):
    N = ROWS * COLS
    f, axes = draw_image_grid(
        x[:N], rows=ROWS, cols=COLS, figsize=(20, 20))
    for i in range(ROWS):
        for j in range(COLS):
            ax = axes[i, j]
            idx = i * COLS + j
            nameof = word_encoder.classes_
            guess = nameof[y_pred[idx]]
            answer = nameof[y_true[idx]]
            ax.set_title(f"Guess: {guess}\nAnswer: {answer}")
            ax.axis('off')
    # plt.show()
    makedirs(output_path)
    f.set_dpi(75)
    f.savefig(output_path)
    plt.close(f)
    print("Save result figure:", output_path)


print(HR)


# ## Load model
model = LitLSTM(
    strokes=MAX_STROKES_LEN,
    out_classes=OUT_CLASSES,
)
model.save_hyperparameters(config)
print("Model Network:")
print(model, HR)

if PREVIOUS_MODEL_STATE:
    print("Use trained model")
    trained_model = torch.load(
        PREVIOUS_MODEL_STATE, map_location=torch.device(device))
    model.load_state_dict(trained_model)
else:
    print("Train model from scratch")


# Logger
wandb_logger = WandbLogger(
    # set the wandb project where this run will be logged
    project="Quick Draw RNN",
    tags=["LSTM", "Adam", "Lightning"],
    # track hyperparameters and run metadata
    config=vars(config),
    log_model=True,
)

lr_find_callback = LearningRateFinder(
    min_lr=1e-6, max_lr=0.1, attr_name="lr",
)
lr_callback = LearningRateMonitor()
checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
earlystop_callback = EarlyStopping(monitor="val_loss", patience=5)


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
        f, axes = draw_image_grid_with_probs(x.detach().cpu().numpy(),
                                             y.detach().cpu().numpy(),
                                             outputs,
                                             y_classes=word_encoder.classes_,
                                             k=5,
                                             rows=6,
                                             cols=6,
                                             figsize=((30, 30)))
        imgs = [wandb.Image(f)]
        wandb_logger.log_image(key="batch_image", images=imgs)
        plt.close(f)


image_preview_callback = SaveBatchImageCallback()


# Trainer
trainer = L.Trainer(
    max_epochs=EPOCH_RUNS,
    max_time="00:05:00:00",
    use_distributed_sampler=True,
    enable_checkpointing=True,
    enable_model_summary=True,
    enable_progress_bar=True,
    log_every_n_steps=5,
    logger=wandb_logger,
    callbacks=[
        lr_callback,
        checkpoint_callback,
        earlystop_callback,
        lr_find_callback,
        image_preview_callback,
    ],
)
dm = SketchesDataModule(batch_size=BATCH_SIZE, encoder=word_encoder)

# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(model, datamodule=dm)
# print("lr_finder:", lr_finder.results)

# fig = lr_finder.plot(suggest=True)
# fig.savefig("lr_find.png")
# plt.close(fig)

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()

# # update hparams of the model
# model.hparams.lr = new_lr

trainer.fit(model, datamodule=dm, ckpt_path="model.ckpt")
trainer.validate(model, dm, verbose=True)


print("Train terminated.")
# wandb.alert(title='Train terminated', text='Check it out')
