import os
import json
import torch
import numpy as np
from typing import Union, List
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
