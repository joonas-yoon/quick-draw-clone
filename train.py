# %% [markdown]
# ## Importings

# %%
from sklearn.metrics import accuracy_score
from typing import Union, List, Optional, Callable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Literal
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch
from tqdm import tqdm, tqdm_notebook
import os
import re
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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
CUDA_GPU_ID = 3

# - Dataset
DATASET_DIR = 'dataset/sketches/sketches'
MAX_STROKES_LEN = 128
TRAIN_SAMPLES_PER_CLASS = 5000
VALID_SAMPLES_PER_CLASS = 100
OUT_CLASSES = 345

# - Training
EPOCH_RUNS = 50
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3

# - Load model
PREVIOUS_MODEL_STATE = None
MODEL_OUTPUT_NAME = f'model_{MAX_STROKES_LEN}_strokes.pt'

# - Logging
LOG_JSON_PATH = "log.json"
FIG_OUTPUT_DIR = 'figures'
LOSS_PLOT_PATH = "plot_train.png"
LOG_SAVE_INTERVAL: int = 1
PLOT_SAVE_INTERVAL: int = 2
MODEL_SAVE_INTERVAL: int = 5


# %%
# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


# %%
if torch.cuda.is_available():
    device = torch.device(
        f'cuda:{CUDA_GPU_ID}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print('device:', device, HR)


# %%
# Helper functions
def makedirs(path: str) -> None:
    if len(path) == 0:
        return None
    dirpath = os.path.dirname(path)
    if dirpath == '':
        dirpath = path
    os.makedirs(dirpath, exist_ok=True)


# %% [markdown]
# ## Prepare to collect dataset

# %%
files = os.listdir(DATASET_DIR)
# files = sorted(list(filter(lambda p: '.full' in p, files)))
files = list(map(lambda p: os.path.join(DATASET_DIR, p), files))


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
    return path.split('/')[-1].split('.')[0]

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

# %%
# word_encoder.transform(['The Eiffel Tower', 'ant'])

# %%


# %%
# ds = np.load(files[0], encoding='latin1', allow_pickle=True)
# ds["train"].shape

# %%
def np_reshape_sequence(stacked_strokes: np.ndarray) -> np.ndarray:
    result = []
    for strokes in stacked_strokes:
        remain = MAX_STROKES_LEN - len(strokes)
        if remain > 0:
            sz = ((0, remain), (0, 0))
            a = np.pad(strokes, sz)
        else:
            a = strokes[:MAX_STROKES_LEN, :]
        result.append(a)
    return np.array(result)

# %%


# %% [markdown]
# ### Load dataset (train/valid)

# %%


def load_datasets(files: list,
                  is_for: Literal["train", "valid", "test"],
                  max_row: int = 0) -> tuple:
    xs = []
    ys = []
    bar = tqdm(total=len(files))
    for file in files:
        basename = filename_to_label(file)
        bar.set_description(f"Load... {str(basename):15s}")
        packed = np.load(file, encoding='latin1', allow_pickle=True)
        pack = packed[is_for]
        rows = len(pack) if max_row == 0 else max_row
        x_data = np_reshape_sequence(pack[:rows])
        xs.append(x_data)
        y_label = basename
        y_class = word_encoder.transform([y_label])[0]
        y_class_reshape = [y_class for _ in range(len(x_data))]
        ys.append(y_class_reshape)
    bar.close()
    return (np.array(xs), np.array(ys))


# %% [markdown]
# ### Load dataset as a class
# %%


class StorkesDataset(Dataset):
    def __init__(self,
                 files: list,
                 is_for: Literal["train", "valid", "test"],
                 max_row: int = 0):
        x, y = load_datasets(files, is_for=is_for, max_row=max_row)
        self.x = x.reshape(-1, MAX_STROKES_LEN, 3)
        self.y = y.reshape(-1)
        self._classes = list(set(map(filename_to_label, files)))
        self._classes_n = len(self._classes)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


# %%
TRAIN_FILES = sorted(files)
print("Files used in dataset", f"({len(TRAIN_FILES)}):")
print('\n'.join(TRAIN_FILES), HR)

# %%
print("Collect dataset to train")
train_dataset = StorkesDataset(
    TRAIN_FILES, "train", max_row=TRAIN_SAMPLES_PER_CLASS)
train_dataset

# %%
# load_datasets(files, "train", max_row=...)
train_x, train_y = train_dataset[:]
print(train_x.shape, train_y.shape)

# %%
print("Collect dataset to valid")
valid_dataset = StorkesDataset(
    TRAIN_FILES, "valid", max_row=VALID_SAMPLES_PER_CLASS)
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


# %% [markdown]
# ## Utils to visualize image

# %%
def weight_to_rgb(weight: float) -> tuple:
    return (
        max(0.0, min(1.0 - 2.0*weight, 1.0)),
        abs(1.0 - abs(2.0*weight - 1.0)),
        max(0.0, min(2.0*weight - 1.0, 1.0))
    )


def rgb_to_hex(rgb: tuple, order: list = [0, 1, 2]) -> str:
    _a = list(map(lambda i: int(255*i), rgb))
    ro, go, bo = order
    r, g, b = _a[ro], _a[go], _a[bo]
    return f"#{r:02x}{g:02x}{b:02x}"


# %%


def get_image_resolution(strokes: list) -> tuple:
    inf = 1e8
    minx, maxx, miny, maxy = inf, -inf, inf, -inf
    x, y = 0, 0
    for stroke in strokes:
        dx, dy, _ = stroke
        x += dx
        y += dy
        minx, maxx = min(x, minx), max(x, maxx)
        miny, maxy = min(y, miny), max(y, maxy)
    return (maxx - minx, maxy - miny)


def reconstruct_to_images(
        rdp_lines,
        size=(256, 256),
        ps=2,
        get_final=False,
        order_color=False
) -> Union[List[Image.Image], Image.Image]:
    rdp_lines = np.array(rdp_lines)
    I_WIDTH, I_HEIGHT = get_image_resolution(rdp_lines)
    I_SHAPE = max(I_WIDTH, I_HEIGHT) * 2.5  # padding
    O_WIDTH, O_HEIGHT = size
    fx, fy = O_WIDTH / I_SHAPE, O_HEIGHT / I_SHAPE
    START_X, START_Y = O_WIDTH // 2, O_HEIGHT // 2
    LINE_WIDTH = int(ps)
#     print(I_WIDTH, I_HEIGHT, O_WIDTH, O_HEIGHT, fx, fy, START_X, START_Y)
    img = Image.new("RGB", (O_WIDTH, O_HEIGHT), "white")
    cx, cy = (START_X, START_Y)
    images = []
    n = len(rdp_lines)
    for i in range(n):
        dx, dy, line_type = rdp_lines[i]
        nx, ny = cx+dx*fx, cy+dy*fy
        is_end = (i-1 >= 0) and (rdp_lines[i-1][2] == 1)
        if not is_end:
            shape = [(cx, cy), (nx, ny)]
            if order_color:
                color = rgb_to_hex(weight_to_rgb(
                    i / n), order=[0, 2, 1])  # r->b->g
            else:
                color = "black"

            draw = ImageDraw.Draw(img)
            draw.line(shape, fill=color, width=LINE_WIDTH)
            if not get_final:
                images.append(img.copy())
        cx, cy = nx, ny
    if get_final:
        return img.copy()
    return images


def reconstruct_to_gif(rdp_lines, filename='output.gif', size=(256, 256), ps=2, order_color=False, **kwargs):
    images = reconstruct_to_images(
        rdp_lines, size=size, ps=ps, order_color=order_color)
    images[0].save(filename, save_all=True, append_images=images[1:], **kwargs)


# %%
print("Image reconstructing test:")


# %%
print(word_encoder.classes_[train_y[7]])
reconstruct_to_images(train_x[7], size=(256, 256), get_final=True)

# %%
print(word_encoder.classes_[train_y[7]],
      "(Coloring storkes by time passed (red->blue->green))")
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


# %%
print(HR)


# %% [markdown]
# ## Define Model network

# %%

# %%
# # Model architecture validation first

# BATCH = 4

# layer1 = nn.Conv1d(in_channels=128, out_channels=48, kernel_size=3)
# lo1 = layer1(torch.zeros((BATCH, 128, 3)))
# print(lo1.shape)

# layer2 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
# lo2 = layer2(lo1)
# print(lo2.shape)

# layer3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=1)
# lo3 = layer3(lo2)
# print(lo3.shape)
# print(lo3.view(BATCH, -1).shape)

# layer4 = nn.LSTM(input_size=96, hidden_size=128, num_layers=2)
# h0 = torch.randn(2, 128)
# c0 = torch.randn(2, 128)
# lo4, (h1, c1) = layer4(lo3.view(BATCH, -1), (h0, c0))
# print(lo4.shape, h1.shape, c1.shape)

# layer5 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2)
# h0 = torch.randn(2, 128)
# c0 = torch.randn(2, 128)
# lo5, (h2, c2) = layer5(lo4, (h1, c1))
# print(lo5.shape, h2.shape, c2.shape)

# layer6 = nn.Linear(in_features=128, out_features=345)
# lo7 = layer6(lo5)
# print(lo7.shape)

# %%

class StrokeRNN(nn.Module):
    def __init__(self, out_classes: int, hidden_state: tuple):
        super().__init__()

        N_STROKES = MAX_STROKES_LEN

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=N_STROKES, out_channels=48, kernel_size=3),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=1),
            nn.Dropout(p=0.3),
        )
        self.lstm1 = nn.LSTM(
            input_size=96, hidden_size=N_STROKES, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=N_STROKES,
                             hidden_size=N_STROKES, num_layers=2)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=N_STROKES, out_features=out_classes),
        )
        self.hidden_state = hidden_state

    def forward(self, x):
        # Conv 1D layer
        x = self.conv(x)
        bs = x.size(0)
        x = x.view(bs, -1)

        # LSTM layer
        x, (h1, c1) = self.lstm1(x, self.hidden_state)
        x = F.dropout(x, p=0.1)
        x, (h2, c2) = self.lstm2(x, (h1, c1))
        x = F.dropout(x, p=0.1)
        x = self.fc(x)

#         # SoftMax if not using CrossEntropyLoss
#         x = F.softmax(x, dim=1)
        return x


# %%


# %% [markdown]
# ### Load previous trained model

# %%
model = StrokeRNN(out_classes=OUT_CLASSES, hidden_state=(
    torch.zeros(2, MAX_STROKES_LEN).to(device),
    torch.zeros(2, MAX_STROKES_LEN).to(device),
))
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5, threshold=1e-3)

# %%


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def check(self, validation_loss) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# %%
early_stopper = EarlyStopper(patience=6)

# %%
# torch.cuda.empty_cache()

# %%
# torch.cuda.is_current_stream_capturing()

# %%


# %%


# %% [markdown]
# ### Run g

# %%
class SyncStream():
    def __init__(self):
        if torch.cuda.is_available():
            self.dev = 'cuda'
        elif torch.backends.mps.is_available():
            self.dev = 'mps'
        else:
            self.dev = 'cpu'

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
            losses.append(_loss)

            y_pred = torch.argmax(log_probs, dim=1)
            acc = accuracy_score(y_pred.cpu(), batch_y.cpu())
            accs.append(acc)

            if cb_batch_end != None:
                cb_batch_end(batch_idx, _loss, acc)

    return np.mean(losses), np.mean(accs)

# %%


# %%
EMPTY_LOGS = {
    "epoch": 0,
    "train_loss": [],
    "train_acc": [],
    "valid_loss": [],
    "valid_acc": [],
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

    bar = tqdm(total=len(train_batchs)+len(valid_batchs), leave=True)

    def when_train_batch_end(_, loss, acc):
        tqdm._instances.clear()
        bar.set_description(f'[train] loss={loss:8.6f} acc={acc*100:4.2}%')
        bar.update(1)

    def when_valid_batch_end(_, loss, acc):
        tqdm._instances.clear()
        bar.set_description(f'[valid] loss={loss:8.6f} acc={acc*100:4.2}%')
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
    logs["train_loss"].append(train_loss)
    logs["train_acc"].append(train_acc)

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
    bar.close()

    logs["valid_loss"].append(valid_loss)
    logs["valid_acc"].append(valid_acc)

    logs["epoch"] = epoch

    print(f'epoch={epoch} | '
          f'train/valid loss={train_loss:8.4f}/{valid_loss:8.4f} | '
          f'train/valid acc={100*train_acc:4.2f}%/{100*valid_acc:4.2f}%')

    scheduler.step()

    # Save log
    if epoch % (LOG_SAVE_INTERVAL or 1) == 0:
        with open(LOG_JSON_PATH, 'w', encoding='utf-8') as fp:
            json.dump(logs, fp=fp, ensure_ascii=True, indent=2)

    # Save plot
    if epoch % (PLOT_SAVE_INTERVAL or 1) == 0:
        save_plot(logs, LOSS_PLOT_PATH)

    # Save model state dict
    if epoch % (MODEL_SAVE_INTERVAL or 1) == 0:
        torch.save(model.state_dict(), f'{MODEL_OUTPUT_NAME}_{epoch}.pt')

    # Early Stop
    if early_stopper.check(validation_loss=valid_loss):
        print("Stopped by early stop")
        break


# %%
print("Train terminated.")
save_plot(logs, LOSS_PLOT_PATH)


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
test_dataset = StorkesDataset(files, "test")
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
                strokes, size=(256, 256), ps=5, get_final=True))
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
