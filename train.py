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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from glob import glob
import gc
gc.enable()

# %%
# Horizontal line as divider
HR = "\n" + ("-" * 30) + "\n"


# %%
# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# %%
if torch.cuda.is_available():
    GPU_ID = 3
    device = torch.device(
        f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print('device:', device, HR)


# %% [markdown]
# ## Prepare to collect dataset

# %%
DATASET_DIR = 'dataset/sketches/sketches'

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
MAX_STROKES_LEN = 128


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
    for file in tqdm(files):
        packed = np.load(file, encoding='latin1', allow_pickle=True)
        pack = packed[is_for]
        rows = len(pack) if max_row == 0 else max_row
        x_data = np_reshape_sequence(pack[:rows])
        xs.append(x_data)
        y_label = filename_to_label(file)
        y_class = word_encoder.transform([y_label])[0]
        y_class_reshape = [y_class for _ in range(len(x_data))]
        ys.append(y_class_reshape)
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
        self.x = x.reshape(-1, 128, 3)
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
TRAIN_FILES = sorted(files[:64])
print("Files used in dataset", f"({len(TRAIN_FILES)}):")
print('\n'.join(TRAIN_FILES), HR)

# %%
print("Collect dataset to train")
train_dataset = StorkesDataset(TRAIN_FILES, "train", max_row=50000)
train_dataset

# %%
# load_datasets(files, "train", max_row=...)
train_x, train_y = train_dataset[:]
print(train_x.shape, train_y.shape)

# %%
print("Collect dataset to valid")
valid_dataset = StorkesDataset(TRAIN_FILES, "valid", max_row=10000)
valid_x, valid_y = valid_dataset[:]
print(valid_x.shape, valid_y.shape, HR)

# %% [markdown]
# ### the number of output classes to train

# %%
n_classes = train_dataset._classes_n
print("# of classes:", n_classes)
print("classes to train in this run:", train_dataset._classes, HR)

# %%


# %% [markdown]
# ### DataLoader & sampler

# %%

# %%
BATCH_SIZE = 256 + 256
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

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=48, kernel_size=3),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=1),
            nn.Dropout(p=0.3),
        )
        self.lstm1 = nn.LSTM(input_size=96, hidden_size=128, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=out_classes),
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
# Use trained model if exists
USE_PREVIOUS_MODEL = True

# %%
if USE_PREVIOUS_MODEL:
    print("Use trained model")
    trained_model = torch.load('model_trained_1.state.pt')
    model = StrokeRNN(out_classes=n_classes, hidden_state=(
        torch.zeros(2, 128).to(device),
        torch.zeros(2, 128).to(device),
    ))
    model.load_state_dict(trained_model)
else:
    print("Create new model to train")
    model = StrokeRNN(out_classes=n_classes, hidden_state=(
        torch.zeros(2, 128).to(device),
        torch.zeros(2, 128).to(device),
    ))

model.to(device)

print(model, HR)

# %% [markdown]
# ### Criterion & Optimizer

# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %%


# %%


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
    cb_batch_end: Optional[Callable[[int, float], None]],
    device: str = 'cpu',
) -> list:
    with SyncStream():
        total_losses = []
        for batch_idx, (x, y) in enumerate(batchs):
            losses = []

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

            if cb_batch_end != None:
                cb_batch_end(batch_idx, np.mean(losses))

        total_losses.append(np.mean(losses))
    return total_losses

# %%


# %%
logs = {
    "epoch": 0,
    "train_loss": [],
    "valid_loss": [],
}

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
    plt.savefig(filename, **kwargs)


# %%
N_CLASS = len(train_y)
EPOCH_RUNS = 0

print("epochs =", EPOCH_RUNS)

model.to(device)

for epoch in range(EPOCH_RUNS):
    bar = tqdm(total=len(train_batchs)+len(valid_batchs), leave=True)

    def when_train_batch_end(_, loss):
        tqdm._instances.clear()
        bar.set_description(f'[train] loss={loss:8.6f}')
        bar.update(1)

    def when_valid_batch_end(_, loss):
        tqdm._instances.clear()
        bar.set_description(f'[valid] loss={loss:8.6f}')
        bar.update(1)

    # Train loop
    model.train()
    train_loss = run_batch(
        model=model,
        batchs=train_batchs,
        criterion=criterion,
        optimizer=optimizer,
        is_train=True,
        cb_batch_end=when_train_batch_end,
        device=device,
    )
    logs["train_loss"].append(np.mean(train_loss))

    # Valid loop
    model.eval()
    valid_loss = run_batch(
        model=model,
        batchs=valid_batchs,
        criterion=criterion,
        optimizer=optimizer,
        is_train=False,
        cb_batch_end=when_valid_batch_end,
        device=device,
    )
    logs["valid_loss"].append(np.mean(valid_loss))

    logs["epoch"] = epoch + int(logs["epoch"] or 0)
    train_loss, valid_loss = logs["train_loss"][-1], logs["valid_loss"][-1]
    print(f'epoch={epoch}, train/valid loss={train_loss:8.4f}/{valid_loss:8.4f}')
    bar.close()

    # Save plot for every 10 epochs
    if epoch % 2 == 0:
        save_plot(logs, "plot_train.png")

    # Save model state dict
    torch.save(model.state_dict(), f'model_trained_{epoch}.state.pt')


# %%
print("Train terminated.")
save_plot(logs, "plot_train.png")


# %%


# %%


# %% [markdown]
# ### Save model

# %%
MODEL_OUTPUT_PATH = 'model_trained.pt'
torch.save(model, MODEL_OUTPUT_PATH)
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
test_batchs = DataLoader(test_dataset, batch_size=128, shuffle=True)
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
    f.savefig(filename)
    plt.close(f)
    print("Save result figure:", filename)


# %%
GRID_SIZE = ROWS * COLS
FIG_OUTPUT_DIR = 'figures'

os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

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
print("accuracy score:", f"{acc*100:6f}%")

# %%


# %%


# %%
