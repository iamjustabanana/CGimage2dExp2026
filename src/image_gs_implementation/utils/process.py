from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """torch [C,H,W] (0~1) -> np.uint8 [H,W,C]（邊界轉換，給 Streamlit 輪播）。"""
    arr = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (arr * 255).round().astype(np.uint8)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """torch [C,H,W] (0~1) -> 存檔(自動 clamp、自動建資料夾)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = to_numpy(tensor)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)
