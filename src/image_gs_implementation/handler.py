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
from . import input_image, gaussians, render, train, utils
# from . import compress


def process(image: np.ndarray, cfg: Config | None = None) -> list[np.ndarray]:
    cfg = cfg or load_config()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Step 1
    target, grid, grad_prob = input_image.process(image, cfg, device)
    _, h, w = target.shape

    # 邊界轉 numpy：原圖 + 梯度圖(拉成 3 通道方便顯示)
    grad_vis = (grad_prob / grad_prob.max()).reshape(h, w).unsqueeze(0).expand(3, h, w)
    results: list[np.ndarray] = [input_image.to_numpy(target), input_image.to_numpy(grad_vis)]

    # 同時存到 config 指定的 out_dir
    utils.save_image(target, f"{cfg.out_dir}/input.png")
    utils.save_image(grad_vis, f"{cfg.out_dir}/gradient.png")

    # ---- Step 2：建立並初始化高斯 ----
    g = gaussians.process(target, grad_prob, grid, cfg, device)
    # 紅點疊在原圖上
    pos_vis = gaussians.visualize_positions(g, target)
    results.append(input_image.to_numpy(pos_vis))
    utils.save_image(pos_vis, f"{cfg.out_dir}/gaussians_init.png")
    # 紅點疊在梯度圖上(不壓暗，方便檢查點是否落在亮邊)
    grad_pos_vis = gaussians.visualize_positions(g, grad_vis, dim=1.0)
    results.append(input_image.to_numpy(grad_pos_vis))
    utils.save_image(grad_pos_vis, f"{cfg.out_dir}/gaussians_on_gradient.png")
    # 紅點疊在全黑背景上
    black = torch.zeros_like(target)
    black_pos_vis = gaussians.visualize_positions(g, black, dim=1.0)
    results.append(input_image.to_numpy(black_pos_vis))
    utils.save_image(black_pos_vis, f"{cfg.out_dir}/gaussians_black.png")

    # ---- Step 3：用(尚未訓練的)初始高斯渲染一張，驗證渲染器(也是訓練前 baseline) ----
    # 預覽渲染不需要梯度，包 no_grad 才不會建龐大計算圖(省記憶體、加速)。
    with torch.no_grad():
        init_render = render.process(g, h, w, grid, cfg)
    results.append(input_image.to_numpy(init_render))
    utils.save_image(init_render, f"{cfg.out_dir}/render_init.png")

    # ---- Step 4(+5)：訓練 ----
    # progressive 開啟時 train 內部會呼叫 Step 5 漸進補高斯；train 內部用 Step 3 render。
    # 注意：純 PyTorch 全量訓練很慢/吃記憶體，高解析請把 config 的 downsample 調大。
    g = train.process(g, target, grid, cfg)

    # ---- Step 3：最終渲染重建圖 ----
    with torch.no_grad():
        pred = render.process(g, h, w, grid, cfg)
    results.append(input_image.to_numpy(pred))
    utils.save_image(pred, f"{cfg.out_dir}/render.png")

    # ---- Step 6：壓縮率 + 量化（待實作）----
    # compress.process(g, h * w, cfg)

    return results
