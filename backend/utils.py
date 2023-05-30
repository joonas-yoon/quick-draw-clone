import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import json
import torch
import numpy as np
from typing import Any, Tuple, Union, List
from PIL import Image, ImageDraw


def get_filename(path: str, ext: bool = True) -> str:
    filename = path.split('/')[-1]
    if not ext:
        return filename.split('.')[0]
    return filename


def makedirs(path: str) -> None:
    if len(path) == 0:
        return None
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    return "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


def weight_to_rgb(weight: float) -> tuple:
    return (
        max(0.0, min(1.0 - 2.0 * weight, 1.0)),
        abs(1.0 - abs(2.0 * weight - 1.0)),
        max(0.0, min(2.0 * weight - 1.0, 1.0))
    )


def rgb_to_hex(rgb: tuple, order: list = [0, 1, 2]) -> str:
    _a = list(map(lambda i: int(255*i), rgb))
    ro, go, bo = order
    r, g, b = _a[ro], _a[go], _a[bo]
    return f"#{r:02x}{g:02x}{b:02x}"


def get_image_box(strokes: np.array) -> tuple:
    points = []
    for i, s in enumerate(strokes):
        dx, dy, _ = s
        if i == 0:
            cx, cy = 0, 0
        cx += dx
        cy += dy
        points += [(cx, cy, _)]
    points = np.array(points)
    xs, ys = points[:, 0], points[:, 1]
    return min(xs), max(xs), min(ys), max(ys)


def get_resolution(strokes: np.array) -> tuple:
    left, right, top, bottom = get_image_box(strokes)
    return right - left, bottom - top


def reconstruct_to_images(
        rdp_lines,
        size=(256, 256),
        ps=2,
        get_final=False,
        order_color=False
) -> Union[List[Image.Image], Image.Image]:
    rdp_lines = np.array(rdp_lines)

    LEFT, RIGHT, TOP, BOTTOM = get_image_box(rdp_lines)
    I_WIDTH, I_HEIGHT = int(RIGHT - LEFT), int(BOTTOM - TOP)
    LINE_WIDTH = int(ps)

    img = Image.new("RGB", (I_WIDTH * 2, I_HEIGHT * 2), "white")
    cx, cy = (I_WIDTH, I_HEIGHT)
    images = []
    n = len(rdp_lines)
    for i in range(n):
        dx, dy, line_type = rdp_lines[i]
        nx, ny = cx + dx, cy + dy
        is_end = (i-1 >= 0) and (rdp_lines[i-1][2] != 0)
        if not is_end:
            shape = [(cx, cy), (nx, ny)]
            if order_color:
                color = rgb_to_hex(
                    weight_to_rgb(i / n),
                    order=[0, 2, 1]
                )  # r->b->g
            else:
                color = "black"

            draw = ImageDraw.Draw(img)
            draw.line(shape, fill=color, width=LINE_WIDTH)
            if not get_final:
                images.append(img.resize(size).copy())
        cx, cy = nx, ny
    if get_final:
        return img.resize(size).copy()
    return images


def reconstruct_to_gif(rdp_lines,
                       filename='output.gif',
                       size=(256, 256),
                       ps=2,
                       order_color=False,
                       **kwargs):
    images = reconstruct_to_images(
        rdp_lines, size=size, ps=ps, order_color=order_color)
    images[0].save(filename, save_all=True, append_images=images[1:], **kwargs)


def draw_image(strokes: np.array, ax):
    cum_arr = np.column_stack(
        (np.cumsum(strokes[:, :2], axis=0), strokes[:, 2]))
    splited = np.split(cum_arr, np.argwhere(cum_arr[:, 2] > 0).flatten()+1)
    n = len(splited)
    for i, polygon in enumerate(splited):
        polygon = polygon[:, 0:2].tolist()
        color = rgb_to_hex(weight_to_rgb(i / n), order=[0, 2, 1])  # r->b->g
        if len(polygon) > 0:
            ax.add_patch(Polygon(polygon, fill=None,
                         closed=False, color=color))
    ax.set_xlim(np.percentile(cum_arr[:, 0], [0, 100]))  # min, max
    ax.set_ylim(np.percentile(cum_arr[:, 1], [100, 0]))  # max, min (fliped)


def draw_image_grid(strokes: np.array, rows: int, cols: int, **kwargs) -> Tuple[plt.figure, np.ndarray]:
    fig, axes = plt.subplots(rows, cols, **kwargs)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for strokes, ax in zip(strokes, axes.flatten()):
        ax.axis('off')
        draw_image(strokes, ax)
    return fig, axes


def top_probs(probs: torch.Tensor, k: int = 5):
    """ probs is (batch, Y)

    Returns:
        (batch, 2, Y)
        which 2 for sorted index, sorted probabilites)
    """
    res = []
    probs = probs.detach().cpu().numpy()
    for p in probs:
        e = np.exp(p)
        norm = e / np.sum(e)
        i = norm.argsort()[-k:]
        res += [(i, norm[i])]
    return np.array(res)


def draw_image_grid_with_probs(
        strokes: np.ndarray,
        y_true: list,
        y_pred_probs: torch.Tensor,
        y_classes: list,
        k: int,
        rows: int = 3,
        cols: int = 3,
        **kwargs,
) -> Tuple[plt.figure, Any]:
    probs = top_probs(y_pred_probs, k=k)
    f, axes = draw_image_grid(strokes, rows, cols, **kwargs)
    axes = axes.flatten()
    for i in range(rows * cols):
        ax = axes[i]
        ans = y_classes[y_true[i]]
        pi, pv = probs[i, 0, :], probs[i, 1, :]
        top3 = [
            f"{y_classes[int(pi[_])]} ({pv[_]*100:6.3f}%)" for _ in range(3)
        ]
        top3 = '\n'.join(top3[::-1])
        ax.set_title(f"Answer:{ans}\n"
                     f"- Predict -\n"
                     f"{top3}")
    return f, axes
