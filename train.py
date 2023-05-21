# %% [markdown]
# ## Importings

# %%
from PIL import Image as PILImage
import wandb
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
    draw_image_grid, get_device_name, get_filename, makedirs, reconstruct_to_images, reconstruct_to_gif
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
SHOW_PROGRESS: bool = config.progress_bar

WB_API: str = config.wandb_api or os.environ['WANDB_API_KEY']

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


# %%
wandb.login(key=WB_API)

# start a new wandb run to track this script
wb_logger = wandb.init(
    # set the wandb project where this run will be logged
    project="Quick Draw RNN",
    tags=["RNN", "LSTM", "Adam", "CosineAnnealingWarmRestarts"],

    # track hyperparameters and run metadata
    config=vars(config)
)


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
print('\n'.join(TRAIN_FILES[:3]))
print('...')
print('\n'.join(TRAIN_FILES[-3:]), HR)

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
save_result_image(
    x=train_x,
    y_true=train_y,
    y_pred=train_y,
    output_path="sample_image_batch.png",
)

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
    trained_model = torch.load(
        PREVIOUS_MODEL_STATE, map_location=torch.device(device))
    model.load_state_dict(trained_model)
else:
    print("Train model from scratch")

# artifact = wandb.Artifact(MODEL_OUTPUT_NAME, type='model')

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
TEMP_BATCH_FILE = "_batch_image.png"


def top_probs(probs: np.ndarray, k: int = 5):
    """ probs is (batch, Y)

    Returns:
        (batch, 2, Y)
        which 2 for sorted index, sorted probabilites)
    """
    res = []
    for p in probs:
        norm = (p - p.min()) / (p.max() - p.min())
        norm = norm / norm.sum()
        i = norm.argsort()[-k:]
        res += [(i, norm[i])]
    return np.array(res)


def draw_image_grid_with_probs(
        strokes: np.ndarray,
        y_true: list,
        y_pred_probs: np.ndarray,
        k: int
):
    probs = top_probs(y_pred_probs, k=k)
    f, axes = draw_image_grid(strokes, ROWS, COLS, figsize=(25, 25))
    axes = axes.flatten()
    for i in range(ROWS * COLS):
        ax = axes[i]
        words = word_encoder.classes_
        ans = words[y_true[i]]
        pi, pv = probs[i, 0, :], probs[i, 1, :]
        top3 = [
            f"{words[int(pi[_])]} ({pv[_]*100:6.3f}%)" for _ in range(3)
        ]
        top3 = '\n'.join(top3[::-1])
        ax.set_title(f"Answer:{ans}\n"
                     f"- Predict -\n"
                     f"{top3}")
    return f, axes


def run_batch(
    model,
    batchs,
    criterion,
    optimizer,
    is_train: bool,
    cb_batch_end: Optional[Callable[[int, float, float], None]],
    logging: bool = False,
    device: str = 'cpu',
) -> tuple:
    losses = []
    accs = []
    with SyncStream():
        for batch_idx, (x, y) in enumerate(batchs):
            strokes: torch.Tensor = torch.as_tensor(
                x).type(torch.FloatTensor).to(device)
            batch_y: torch.Tensor = torch.as_tensor(
                y).type(torch.LongTensor).to(device)

            log_probs: torch.Tensor = model(strokes)
            loss = criterion(log_probs, batch_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _loss = loss.item()

            losses.append(_loss)

            batch_y = batch_y.cpu()
            y_pred = torch.argmax(log_probs, dim=1).cpu()
            acc = accuracy_score(y_pred, batch_y)
            accs.append(acc)

            if cb_batch_end != None:
                cb_batch_end(batch_idx, _loss, acc)

            if logging:
                wandb.log({
                    'batch_loss': _loss,
                    'batch_acc': acc
                })
                if batch_idx == 0:
                    strokes: np.ndarray = strokes.detach().cpu().numpy()
                    probs = log_probs.detach().cpu().numpy()

                    f, axes = draw_image_grid_with_probs(
                        strokes, batch_y, probs, k=3
                    )
                    img = wandb.Image(f)
                    wandb.log({'batch_image': img})
                    plt.close(f)

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

    if SHOW_PROGRESS:
        bar = tqdm(total=len(train_batchs)+len(valid_batchs), leave=True)

    def when_train_batch_end(_, loss, acc):
        if not SHOW_PROGRESS:
            return
        tqdm._instances.clear()
        bar.set_description(f'[train] loss={loss:8.6f} | '
                            f'acc={acc:8.6f} | '
                            f'lr={lr:8.6f}')
        bar.update(1)

    def when_valid_batch_end(_, loss, acc):
        if not SHOW_PROGRESS:
            return
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
        logging=True,
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

    if SHOW_PROGRESS:
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

    wandb.log({
        "lr": lr,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_acc_top_5": train_acc_top_5,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_acc_top_5": valid_acc_top_5,
    })

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
        path = f'{MODEL_OUTPUT_NAME}_{epoch}.pt'
        torch.save(model.state_dict(), path)
        # artifact.add_file(path, is_tmp=True)
        # wb_logger.log_artifact(artifact)

    # Early Stop
    if early_stopper.check(validation_loss=valid_loss):
        print("Stopped by early stop")
        break


# %%
print("Train terminated.")
save_plot(logs, PLOT_PATH)
wandb.alert(title='Train terminated', text='Check it out')


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
n_pages = min(10, len(train_batchs))

for batch_idx, (test_x, y_true) in enumerate(train_batchs):
    if batch_idx >= n_pages:
        break

    test_batch_x = torch.as_tensor(test_x).type(torch.FloatTensor).to(device)
    test_batch_y = y_true.detach().cpu().numpy()

    log_probs = model(test_batch_x)
    y_pred = np.argmax(log_probs.detach().cpu(), axis=1)

    filename = f"test_sample_batch_{batch_idx}.png"

    save_result_image(
        x=test_x,
        y_true=y_true,
        y_pred=y_pred,
        output_path=filename,
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

    batchs = tqdm(test_batchs) if SHOW_PROGRESS else test_batchs

    for (x, y) in batchs:
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
wb_logger.finish()

# %%


# %%
