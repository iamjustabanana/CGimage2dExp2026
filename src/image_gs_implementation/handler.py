"""
Pipeline 總指揮
===============
對外只暴露 process()，給 Streamlit / __main__ 呼叫。
慣例：邊界用 numpy(np.uint8 RGB)，內部用 torch tensor 跑核心。

handler 本身很薄 —— 只負責「依序呼叫每個 step 的 process()」+「邊界 numpy 轉換」，
每一步的實作細節都在各自子套件裡：
  Step 1  input_image.process : numpy -> (target, grid, grad_prob)
  Step 2  gaussians.process   : 建立並初始化高斯
  Step 3  render.process      : 高斯 -> 圖
  Step 4  train.process       : 最佳化高斯逼近原圖
  Step 5  compress.process    : 壓縮率 + 量化

目前 Step 2~5 尚未實作；先回傳 [原圖, 梯度圖] 讓前端能跑，
完成一步就把對應段落打開、把重建圖 append 進結果。
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config, load_config, resolve_device
from . import input_image
# from . import gaussians, render, train, compress


def process(image: np.ndarray, cfg: Config | None = None) -> list[np.ndarray]:
    cfg = cfg or load_config()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)

    # Step 1
    target, grid, grad_prob = input_image.process(image, cfg, device)
    _, h, w = target.shape

    # 邊界轉 numpy：原圖 + 梯度圖(拉成 3 通道方便顯示)
    grad_vis = (grad_prob / grad_prob.max()).reshape(h, w).unsqueeze(0).expand(3, h, w)
    results: list[np.ndarray] = [input_image.to_numpy(target), input_image.to_numpy(grad_vis)]

    # 同時存到 config 指定的 out_dir
    input_image.save_image(target, f"{cfg.out_dir}/input.png")
    input_image.save_image(grad_vis, f"{cfg.out_dir}/gradient.png")

    # ---- Step 2~5：待實作（完成後逐段打開）----
    # g = gaussians.process(target, grad_prob, grid, cfg, device)
    # compress.process(g, h * w, cfg)
    # g = train.process(g, target, grid, cfg)
    # pred = render.process(g, h, w, grid, cfg)
    # results.append(input_image.to_numpy(pred))

    return results
