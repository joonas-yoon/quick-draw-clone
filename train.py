# %% [markdown]
# ## Importings

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from typing import Optional, Callable

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from models.SimpleLSTM import SimpleLSTM, SimpleLSTMBn
from trainer.callbacks import EarlyStopper
from trainer.dataloader import StrokesDataset
from utils import (
    get_device_name, get_filename, makedirs, reconstruct_to_images, reconstruct_to_gif
)

from args import args as config

from glob import glob
import gc
gc.enable()

print("pytorch version:", torch.__version__)


# %%
# Horizontal line as divider
HR = "\n" + ("-" * 30) + "\n"

# %%
# Options
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

# %%
# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


# %%
device = get_device_name()

if torch.cuda.is_available():
    device = f'cuda:{CUDA_GPU_ID}'
    torch.cuda.set_device(device)

print('device:', device, HR)


# %% [markdown]
# ## Prepare to collect dataset

# %%
files = os.listdir(DATASET_DIR)
files = sorted(list(map(lambda p: os.path.join(DATASET_DIR, p), files)))


# %%
print("dataset shape of train/valid/test (sample)")
for filepath in files:
    npz = np.load(filepath, encoding='latin1', allow_pickle=True)
    print(filepath, npz["train"].shape, npz["valid"].shape, npz["test"].shape)
    break
print(HR)

# %%


# %%
def filename_to_label(path: str) -> str:
    return get_filename(path, ext=False)

# %% [markdown]
# ### Label encoding


# %%
labels = list(map(filename_to_label, files))
labels[:10]

# %%
word_encoder = LabelEncoder()
word_encoder.fit(labels)
print('target words for output:', len(word_encoder.classes_), '=>',
      ', '.join([x for x in word_encoder.classes_]), HR)


# %% [markdown]
# ### Load dataset (train/valid)


# %%
TRAIN_FILES = sorted(files)
print("Files used in dataset", f"({len(TRAIN_FILES)}):")
print('\n'.join(TRAIN_FILES), HR)

# %%
print("Collect dataset to train")
train_dataset = StrokesDataset(
    TRAIN_FILES,
    "train",
    max_row=TRAIN_SAMPLES_PER_CLASS,
    strokes=MAX_STROKES_LEN,
    # storke_normalize=True,
    y_transform=word_encoder)

# %%
# load_datasets(files, "train", max_row=...)
train_x, train_y = train_dataset[:]
print(train_x.shape, train_y.shape, '\n')

# %%
print("Collect dataset to valid")
valid_dataset = StrokesDataset(
    TRAIN_FILES,
    "valid",
    max_row=VALID_SAMPLES_PER_CLASS,
    strokes=MAX_STROKES_LEN,
    # storke_normalize=True,
    y_transform=word_encoder)
valid_x, valid_y = valid_dataset[:]
print(valid_x.shape, valid_y.shape, HR)

# %% [markdown]
# ### the number of output classes to train

# %%
print("# of classes:", OUT_CLASSES)
print("classes to train in this run: ", train_dataset._classes_n)
print(train_dataset._classes, HR)

# %%


# %% [markdown]
# ### DataLoader & sampler

# %%

# %%
print("BATCH_SIZE =", BATCH_SIZE, HR)

# %%
# train set is very ordered in y labels like "[0, 0, ..., 1, 1, ..., n, n]"
# so we have to do shuffle on it
train_sampler = SubsetRandomSampler(range(len(train_dataset)))

# %%
train_batchs = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
train_x, train_y = next(iter(train_batchs))
# print(train_x, train_y)

# %%
valid_batchs = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
valid_x, valid_y = next(iter(valid_batchs))
# print(valid_x, valid_y)

# %%


# %%
print("Image reconstructing test:")


# %%
print(word_encoder.classes_[train_y[7]])
reconstruct_to_images(train_x[7], size=(256, 256), get_final=True)

# %%
print(word_encoder.classes_[train_y[7]],
      "(Coloring strokes by time passed (red->blue->green))")
reconstruct_to_images(train_x[7], size=(
    256, 256), get_final=True, ps=6, order_color=True)

# %%
reconstruct_to_gif(train_x[7], filename='sample.gif',
                   duration=66, ps=5, size=(256, 256), loop=0)
reconstruct_to_gif(train_x[7], filename='sample_color.gif',
                   duration=66, ps=5, size=(256, 256), loop=0, order_color=True)

# %% [markdown]
# ![](sample.gif) ![](sample_color.gif)

# %%
print(HR)


# %% [markdown]
# ## Load model

# %%
model = SimpleLSTM(
    strokes=MAX_STROKES_LEN,
    out_classes=OUT_CLASSES,
    device=device,
)
print("Model Network:")
print(model, HR)

if PREVIOUS_MODEL_STATE:
    print("Use trained model")
    trained_model = torch.load(PREVIOUS_MODEL_STATE)
    model.load_state_dict(trained_model)
else:
    print("Train model from scratch")


# %% [markdown]
# ### Criterion & Optimizer

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.8, patience=3, threshold=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer, T_0=10, T_mult=1, eta_min=1e-5)

# %%


# %%
early_stopper = EarlyStopper(patience=10, threshold=1e-2)


# %% [markdown]
# ### Run g

# %%
class SyncStream():
    def __init__(self):
        self.dev: str = get_device_name()

    def __enter__(self):
        if self.dev == 'cuda':
            stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            return torch.cuda.stream(stream)
        else:
            return self

    def __exit__(self, type, value, traceback):
        pass


# %%

def run_batch(
    model,
    batchs,
    criterion,
    optimizer,
    is_train: bool,
    cb_batch_end: Optional[Callable[[int, float, float], None]],
    device: str = 'cpu',
) -> tuple:
    losses = []
    accs = []
    with SyncStream():
        for batch_idx, (x, y) in enumerate(batchs):
            batch_x = torch.as_tensor(x).type(torch.FloatTensor).to(device)
            batch_y = torch.as_tensor(y).type(torch.LongTensor).to(device)

            log_probs = model(batch_x)
            loss = criterion(log_probs, batch_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _loss = loss.item()

            if np.isnan(_loss):
                print(x)
                print(x.shape)
                print(batch_x)
                print(batch_x.shape)
                print(batch_y)
                print(batch_y.shape)
                print(loss)
                print(log_probs)
                exit(0)

            losses.append(_loss)

            y_pred = torch.argmax(log_probs, dim=1)
            acc = accuracy_score(y_pred.cpu(), batch_y.cpu())
            accs.append(acc)

            if cb_batch_end != None:
                cb_batch_end(batch_idx, _loss, acc)

    return losses, accs

# %%


# %%
EMPTY_LOGS = {
    "epoch": 0,
    "train_loss": [],
    "train_acc": [],
    "train_acc_top_5": [],
    "valid_loss": [],
    "valid_acc": [],
    "valid_acc_top_5": [],
    "lr": [],
}

# %%
# Load log json and validate
try:
    keys = set(EMPTY_LOGS.keys())
    with open(LOG_JSON_PATH, mode='r', encoding='utf-8') as fp:
        logs = json.load(fp=fp)

    loaded_keys = set(logs.keys())
    if len(keys - loaded_keys) > 0:
        raise Exception("Invalid log json format")
except Exception as err:
    print(err)
    print("Start logging from empty")
    logs = dict(EMPTY_LOGS)

print(HR)

# %%


def save_plot(logs: dict, filename: str, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    x = list(range(len(logs["train_loss"])))
    ax.plot(x, logs["train_loss"], label='train loss')
    ax.plot(x, logs["valid_loss"], label='valid loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title("loss/epoch")
    ax.legend()
    makedirs(filename)
    fig.savefig(filename, **kwargs)
    plt.close(fig)


# %%
print("epochs =", EPOCH_RUNS)
PREV_EPOCHS = logs["epoch"] or 0

model.to(device)

for epoch_idx in range(EPOCH_RUNS):
    epoch = 1 + PREV_EPOCHS + epoch_idx
    lr = optimizer.param_groups[0]['lr']

    bar = tqdm(total=len(train_batchs)+len(valid_batchs), leave=True)

    def when_train_batch_end(_, loss, acc):
        tqdm._instances.clear()
        bar.set_description(f'[train] loss={loss:8.6f} | '
                            f'acc={acc:8.6f} | '
                            f'lr={lr:8.6f}')
        bar.update(1)

    def when_valid_batch_end(_, loss, acc):
        tqdm._instances.clear()
        bar.set_description(f'[valid] loss={loss:8.6f} | '
                            f'acc={acc:8.6f} | '
                            f'lr={lr:8.6f}')
        bar.update(1)

    # Train loop
    model.train()
    train_loss, train_acc = run_batch(
        model=model,
        batchs=train_batchs,
        criterion=criterion,
        optimizer=optimizer,
        is_train=True,
        cb_batch_end=when_train_batch_end,
        device=device,
    )
    train_loss = np.mean(train_loss)
    train_acc_top_5 = np.mean(sorted(train_acc)[-5:])
    train_acc = np.mean(train_acc)

    logs["train_loss"].append(train_loss)
    logs["train_acc"].append(train_acc)
    logs["train_acc_top_5"].append(train_acc_top_5)

    # Valid loop
    model.eval()
    valid_loss, valid_acc = run_batch(
        model=model,
        batchs=valid_batchs,
        criterion=criterion,
        optimizer=optimizer,
        is_train=False,
        cb_batch_end=when_valid_batch_end,
        device=device,
    )
    valid_loss = np.mean(valid_loss)
    valid_acc_top_5 = np.mean(sorted(valid_acc)[-5:])
    valid_acc = np.mean(valid_acc)

    bar.close()

    logs["valid_loss"].append(valid_loss)
    logs["valid_acc"].append(valid_acc)
    logs["valid_acc_top_5"].append(valid_acc_top_5)
    logs["lr"].append(lr)

    logs["epoch"] = epoch

    print(f'epoch={epoch} | '
          f'train/valid loss={train_loss:6.4f}/{valid_loss:6.4f} | '
          f'train/valid acc={100*train_acc:6.3f}%/{100*valid_acc:6.3f}% | '
          f'train/valid acc (top 5)={100*train_acc_top_5:6.3f}%/{100*valid_acc_top_5:6.3f}%')

    scheduler.step()

    # Save log
    if epoch % (LOG_SAVE_INTERVAL or 1) == 0:
        with open(LOG_JSON_PATH, 'w', encoding='utf-8') as fp:
            json.dump(logs, fp=fp, ensure_ascii=True, indent=2)

    # Save plot
    if epoch % (PLOT_SAVE_INTERVAL or 1) == 0:
        save_plot(logs, PLOT_PATH)

    # Save model state dict
    if epoch % (MODEL_SAVE_INTERVAL or 1) == 0:
        torch.save(model.state_dict(), f'{MODEL_OUTPUT_NAME}_{epoch}.pt')

    # Early Stop
    if early_stopper.check(validation_loss=valid_loss):
        print("Stopped by early stop")
        break


# %%
print("Train terminated.")
save_plot(logs, PLOT_PATH)


# %%


# %%


# %% [markdown]
# ### Save model

# %%
MODEL_OUTPUT_PATH = f'{MODEL_OUTPUT_NAME}.pt'
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
print("Model saved into file:", MODEL_OUTPUT_PATH, HR)

# %%


# %%


# %% [markdown]
# ## Score & result

# %% [markdown]
# ### Load dataset for test

# %%
print("Load dataset to test")
test_dataset = StrokesDataset(
    files,
    "test",
    strokes=MAX_STROKES_LEN,
    y_transform=word_encoder)
test_batchs = DataLoader(test_dataset, batch_size=BATCH_SIZE)
test_x, y_true = next(iter(test_batchs))
print(test_x.shape, y_true.shape)

# %%
print(word_encoder.classes_[y_true[2]])
reconstruct_to_images(test_x[2].detach().cpu().numpy().astype(
    int), size=(256, 256), get_final=True)

# %%
model.eval()

test_batch_x = torch.as_tensor(test_x).type(torch.FloatTensor).to(device)
test_batch_y = y_true.detach().cpu().numpy()

log_probs = model(test_batch_x)
test_y_pred = np.argmax(log_probs.detach().cpu(), axis=1)

# %%
ROWS = 6
COLS = 6


def save_result_image(
    x: list,
    y_true: list,
    y_pred: list,
    filename: str,
):
    f, axes = plt.subplots(ROWS, COLS, figsize=(30, 30))
    for i in range(ROWS):
        for j in range(COLS):
            ax = axes[i, j]
            idx = i * COLS + j
            nameof = word_encoder.classes_
            guess = nameof[y_pred[idx]]
            answer = nameof[y_true[idx]]
            strokes = x[idx].detach().cpu().numpy().astype(int)

            ax.set_title(f"Guess: {guess}\nAnswer: {answer}")
            ax.imshow(reconstruct_to_images(
                strokes, size=(512, 512), ps=5, get_final=True, order_color=True))
            ax.axis('off')
    # plt.show()
    makedirs(filename)
    f.savefig(filename)
    plt.close(f)
    print("Save result figure:", filename)


# %%
GRID_SIZE = ROWS * COLS

n_pages = min(10, len(train_batchs))

for batch_idx, (test_x, y_true) in enumerate(train_batchs):
    if batch_idx >= n_pages:
        break

    test_batch_x = torch.as_tensor(test_x).type(torch.FloatTensor).to(device)
    test_batch_y = y_true.detach().cpu().numpy()

    log_probs = model(test_batch_x)
    y_pred = np.argmax(log_probs.detach().cpu(), axis=1)

    filename = f"test_sample_batch_{batch_idx}.png"
    output_path = os.path.join(FIG_OUTPUT_DIR, filename)

    save_result_image(
        x=test_x,
        y_true=y_true,
        y_pred=y_pred,
        filename=output_path
    )

print(HR)

# %%

# %%
print("Release memory before scoring")
del train_dataset
del valid_dataset
gc.collect()


# %%
print("Scoring with test dataset:")

with SyncStream():
    test_y_trues = []
    test_y_preds = []

    for (x, y) in tqdm(test_batchs):
        batch_x = torch.as_tensor(x).type(torch.FloatTensor).to(device)

        log_probs = model(batch_x)
        test_y_pred = np.argmax(log_probs.detach().cpu(), axis=1)

        test_y_trues += list(y)
        test_y_preds += list(test_y_pred)

print(HR)

# %%

acc = accuracy_score(test_y_trues, test_y_preds)
print("accuracy score:", f"{acc*100:8.6f}%")

# %%


# %%


# %%
