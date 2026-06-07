from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """torch [C,H,W] (0~1) -> np.uint8 [H,W,C]（邊界轉換，給 Streamlit 輪播）。"""
    arr = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return (arr * 255).round().astype(np.uint8)


def error_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel L1 誤差套 Magma colormap（對齊 image-gs applyMagma=True 風格）。

    深紫 = 誤差小；亮黃 = 誤差大。回傳 [3,H,W] float32 in [0,1]。
    """
    import matplotlib.cm as cm                                       # streamlit 已拉進 venv
    err = (pred - target).abs().mean(dim=0).detach().cpu().numpy()  # [H,W] in [0,1]
    colored = cm.magma(err)[:, :, :3].astype("float32")             # [H,W,3] RGB
    return torch.from_numpy(colored).permute(2, 0, 1)               # [3,H,W]


def save_image(tensor: torch.Tensor, path: str) -> None:
    """torch [C,H,W] (0~1) -> 存檔(自動 clamp、自動建資料夾)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = to_numpy(tensor)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)
