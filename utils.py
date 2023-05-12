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


def reconstruct_to_gif(rdp_lines,
                       filename='output.gif',
                       size=(256, 256),
                       ps=2,
                       order_color=False,
                       **kwargs):
    images = reconstruct_to_images(
        rdp_lines, size=size, ps=ps, order_color=order_color)
    images[0].save(filename, save_all=True, append_images=images[1:], **kwargs)
