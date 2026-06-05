from __future__ import annotations

import os

from PIL import Image

from ..input_image import to_numpy


def save_image(tensor, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = to_numpy(tensor)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)
