import os
import numpy as np

from typing import Literal
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import get_filename


def pad_sequence(stacked_strokes: np.ndarray, pad: int = 128) -> np.ndarray:
    result = []
    for strokes in stacked_strokes:
        remain = pad - len(strokes)
        if remain > 0:
            sz = ((0, remain), (0, 0))
            a = np.pad(strokes, sz)
        else:
            a = strokes[:pad, :]
        result.append(a)
    return np.array(result)


def _filename_to_label(filename: str) -> str:
    return get_filename(filename, ext=False)


def load_datasets(files: list,
                  is_for: Literal["train", "valid", "test"],
                  y_transform: LabelEncoder,
                  strokes: int = 128,
                  max_row: int = 0) -> tuple:
    xs = []
    ys = []
    bar = tqdm(total=len(files))
    for file in files:
        basename = _filename_to_label(file)
        bar.set_description(f"Load... {str(basename):15s}")
        packed = np.load(file, encoding='latin1', allow_pickle=True)
        pack = packed[is_for]
        rows = len(pack) if max_row == 0 else max_row
        x_data = pad_sequence(pack[:rows], pad=strokes)
        xs.append(x_data)
        y_label = basename
        y_class = y_transform.transform([y_label])[0]
        y_class_reshape = [y_class for _ in range(len(x_data))]
        ys.append(y_class_reshape)
        bar.update(1)
    bar.close()
    return (np.array(xs), np.array(ys))


class StrokesDataset(Dataset):
    def __init__(self,
                 files: list,
                 is_for: Literal["train", "valid", "test"],
                 y_transform: LabelEncoder,
                 strokes: int = 128,
                 max_row: int = 0):
        x, y = load_datasets(
            files,
            is_for=is_for,
            y_transform=y_transform,
            strokes=strokes,
            max_row=max_row
        )
        self.x = x.reshape(-1, strokes, 3)
        self.y = y.reshape(-1)
        self._classes = list(set(map(_filename_to_label, files)))
        self._classes_n = len(self._classes)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
