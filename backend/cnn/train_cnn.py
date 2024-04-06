
# ## Importings
from datamodules import QuickDrawDataAllSet, DataModule
from typing import Tuple
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, LearningRateFinder, Callback
)
from lightning.pytorch.loggers import WandbLogger
import warnings
from lightning.pytorch.tuner import Tuner
from PIL import Image as PILImage
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

import torch
from torchvision import transforms as T
from tqdm import tqdm
import lightning as L

import models

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

# ## Load model
model = models.CNNModel(len(dataset_all.get_labels()), dropout=config.dropout)
model.to(device)
model.save_hyperparameters(config)
print("Model: ===============================")
print(model, HR)

wandb_logger.watch(model, log="all", log_graph=True)
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
