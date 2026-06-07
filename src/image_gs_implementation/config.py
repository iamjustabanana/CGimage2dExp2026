"""
設定載入
========
真正的參數值放在同目錄的 config.yaml；這支負責把它讀成一個 Config 物件，
讓各模組可以用點記法存取（cfg.num_gaussians）。

  load_config(path=None) : 讀 yaml -> Config（不給 path 就用預設 config.yaml）
  resolve_device(device) : "auto"/None -> 依序 cuda > mps > cpu

仍用 dataclass 是為了「欄位有預設值 + 型別提示」；yaml 缺的欄位會自動沿用預設。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml

DEFAULT_YAML = Path(__file__).with_name("config.yaml")


@dataclass
class Config:
    # ---- 輸入影像（Step 1）----
    downsample: int = 4
    # ---- 高斯（Step 2）----
    num_gaussians: int = 5000
    init_mode: str = "gradient"
    init_random_ratio: float = 0.3
    init_scale: float = 5.0
    inverse_scale: bool = True   # 論文預設：參數存 1/s(inverse scale)，收斂更快更好(Eq.2 footnote)
    # ---- 渲染（Step 3）----
    topk: int = 10               # 每像素只取最近 K 個高斯(論文做法)；0 = 全量(較慢)
    tile_size: int = 64          # >0：tile 渲染(切方塊+剔除遠處高斯，快)；0：全量 all-pairs(慢)。純PyTorch最佳~64
    # ---- 訓練（Step 4）----
    max_steps: int = 5000
    pos_lr: float = 0.3
    scale_lr: float = 2e-3
    rot_lr: float = 2e-3
    feat_lr: float = 5e-3
    ssim_loss_ratio: float = 0.1
    eval_steps: int = 100
    save_image_steps: int = 500
    save_on_add: bool = True       # 每次 progressive 補高斯後也存一張
    # lr 衰減 / 早停（Step 4）
    lr_schedule: bool = True
    decay_ratio: float = 10.0
    check_decay_steps: int = 1000
    max_decay_times: int = 1
    decay_threshold: float = 1e-3
    # ---- 漸進最佳化（Step 5）----
    progressive: bool = True
    initial_ratio: float = 0.5   # 一開始只放總數的這個比例
    add_times: int = 4           # 分幾次把高斯加到 num_gaussians
    add_steps: int = 500         # 每隔幾步加一次
    post_min_steps: int = 3000   # 加滿後至少再訓練幾步
    # ---- 量化／壓縮（Step 6）----
    quantize: bool = False
    pos_bits: int = 16
    scale_bits: int = 16
    rot_bits: int = 16
    feat_bits: int = 16
    # ---- 裝置／輸出 ----
    device: str = "auto"
    out_dir: str = "src/image_gs_implementation/outputs"
    seed: int = 123


def load_config(path: str | Path | None = None) -> Config:
    """讀 yaml -> Config。yaml 沒寫到的欄位沿用 dataclass 預設；多餘的鍵會被忽略。"""
    path = Path(path) if path else DEFAULT_YAML
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in valid})


def resolve_device(device: str | None) -> str:
    """把 device 解析成實際可用的裝置。

    給定明確值(例如 "cuda"/"cpu")就直接用；給 None 或 "auto" 則依序偵測：
        cuda(NVIDIA) > mps(Apple) > cpu
    """
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
