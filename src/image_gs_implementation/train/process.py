"""
Step 4: 訓練迴圈  ✅ 已完成
==========================
用梯度下降把高斯逼近目標圖（這就是「把圖片轉成高斯」的過程）。

每一步：
  pred = render.process(gaussians)            # Step 3 渲染
  loss = L1(pred, target) + ratio·(1-SSIM)    # 差多少
  loss.backward(); optimizer.step()           # autograd 算梯度 -> 微調高斯
  clip：θ∈[0,π]、scale>0(論文 Eq.2 維護有效範圍)
  progressive(Step 5)：每隔 add_steps 步、還沒加滿 -> 補一批高斯並「重建 optimizer」
  lr schedule：PSNR 沒進步就把 lr 除以 decay_ratio；衰減超過上限 -> 早停

要點：不同參數不同 lr(Adam param_groups)。SSIM 用純 PyTorch 簡版(高斯窗)。
對應官方：image-gs/model.py optimize / _get_total_loss / _init_optimization / _lr_schedule
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import render
from . import progressive
from ..input_image import psnr
from ..utils import save_image


def make_optimizer(gaussians, cfg):
    """建 Adam，對 xy/scale/rot/feat 各設一組 lr(論文：位置/尺度/旋轉/顏色學習率不同)。"""
    return torch.optim.Adam([
        {"params": [gaussians.xy], "lr": cfg.pos_lr},
        {"params": [gaussians.scale], "lr": cfg.scale_lr},
        {"params": [gaussians.rot], "lr": cfg.rot_lr},
        {"params": [gaussians.feat], "lr": cfg.feat_lr},
    ])


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    """1D 高斯權重(給 SSIM 的窗用)，總和=1。"""
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """純 PyTorch SSIM(結構相似度，越接近 1 越像)。x,y: [C,H,W] 值域 0~1。"""
    C = x.shape[0]
    g1 = _gaussian_window(window_size, sigma, x.device)
    kernel = (g1[:, None] * g1[None, :]).expand(C, 1, window_size, window_size)  # [C,1,k,k]
    pad = window_size // 2
    xb, yb = x.unsqueeze(0), y.unsqueeze(0)                      # [1,C,H,W]
    # 局部均值(用高斯窗卷積)
    mu_x = F.conv2d(xb, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(yb, kernel, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    # 局部變異數/共變異數
    sig_x2 = F.conv2d(xb * xb, kernel, padding=pad, groups=C) - mu_x2
    sig_y2 = F.conv2d(yb * yb, kernel, padding=pad, groups=C) - mu_y2
    sig_xy = F.conv2d(xb * yb, kernel, padding=pad, groups=C) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2))
    return ssim_map.mean()


def _clip_params(gaussians) -> None:
    """維護論文 Eq.2 的有效範圍：旋轉角 θ∈[0,π]、scale 保持正。"""
    with torch.no_grad():
        gaussians.rot.clamp_(0.0, math.pi)
        gaussians.scale.clamp_(min=1e-3)        # 正值(inverse_scale 時 = 1/s>0)


def train(gaussians, target, grid, cfg):
    """Step 4 主函式：訓練並回傳訓練好的 gaussians(progressive 開時內部呼叫 Step 5)。"""
    _, h, w = target.shape
    device = target.device
    optimizer = make_optimizer(gaussians, cfg)

    # progressive 需要足夠步數把高斯補滿後再收斂，必要時自動拉高 max_steps
    max_steps = cfg.max_steps
    if cfg.progressive:
        need = cfg.add_steps * cfg.add_times + cfg.post_min_steps
        max_steps = max(max_steps, need)

    # 每次訓練建一個乾淨的 steps 子資料夾，舊的先清掉
    steps_dir = os.path.join(cfg.out_dir, "steps")
    if os.path.isdir(steps_dir):
        for f in os.listdir(steps_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(steps_dir, f))
    os.makedirs(steps_dir, exist_ok=True)

    best_psnr, no_improve, decays = 0.0, 0, 0
    early_stop = False

    pbar = tqdm(range(1, max_steps + 1), desc="Training", unit="step")
    for step in pbar:
        # --- 渲染 + loss + 反傳 + 更新 ---
        pred = render.process(gaussians, h, w, grid, cfg)        # Step 3
        loss = F.l1_loss(pred, target)                           # L1
        if cfg.ssim_loss_ratio > 0:
            loss = loss + cfg.ssim_loss_ratio * (1.0 - ssim(pred, target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _clip_params(gaussians)

        # --- Step 5：漸進補高斯(還沒加滿時，每 add_steps 步補一批) ---
        if cfg.progressive and step % cfg.add_steps == 0 and gaussians.num_gaussians < cfg.num_gaussians:
            add = progressive.num_to_add(gaussians.num_gaussians, cfg)
            if add > 0:
                gaussians = progressive.process(gaussians, target, grid, cfg, device, add)
                optimizer = make_optimizer(gaussians, cfg)       # 參數換新 -> optimizer 重建
                pbar.write(f"[step {step}] +{add} gaussians -> {gaussians.num_gaussians}")

        # --- 評估 + lr 衰減/早停 ---
        if step % cfg.eval_steps == 0:
            with torch.no_grad():
                cur = psnr(pred.clamp(0, 1), target)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{cur:.2f}", N=gaussians.num_gaussians)
            # 只有在高斯加滿後才開始 lr 排程/早停
            if cfg.lr_schedule and gaussians.num_gaussians >= cfg.num_gaussians:
                if cur > best_psnr + cfg.decay_threshold:
                    best_psnr, no_improve = cur, 0
                else:
                    no_improve += cfg.eval_steps
                    if no_improve >= cfg.check_decay_steps:
                        no_improve, decays = 0, decays + 1
                        if decays > cfg.max_decay_times:
                            pbar.write(f"[step {step}] early stop (no improvement)")
                            early_stop = True
                            break
                        for pg in optimizer.param_groups:
                            pg["lr"] /= cfg.decay_ratio
                        pbar.write(f"[step {step}] lr decayed /{cfg.decay_ratio}")

        if cfg.save_image_steps and step % cfg.save_image_steps == 0:
            with torch.no_grad():
                cur = psnr(pred.clamp(0, 1), target)
            save_image(pred, os.path.join(steps_dir, f"step{step:05d}_psnr{cur:.1f}.png"))

    pbar.close()
    if not early_stop:
        pbar.write(f"Training completed: {max_steps} steps")

    return gaussians


def process(gaussians, target, grid, cfg):
    """別名：與其他 step 一致的進入點(等同 train)。"""
    return train(gaussians, target, grid, cfg)


# 自我驗證：uv run python -m src.image_gs_implementation.train.process
# 在小圖上訓練幾百步，PSNR 應一路上升、輸出越來越像原圖。
if __name__ == "__main__":
    import sys
    import time
    import numpy as np
    from PIL import Image
    from ..config import load_config, resolve_device
    from ..gaussians import process as build_gaussians
    from ..input_image import to_tensor, get_grid, gradient_map

    cfg = load_config()
    device = resolve_device(cfg.device)
    # 為了快又省記憶體：縮小圖、減步數(全量 all-pairs+autograd 很吃記憶體)
    cfg.downsample = 8            # 2k -> 256px
    cfg.num_gaussians = 2000
    cfg.max_steps = 800
    cfg.add_steps = 150
    cfg.post_min_steps = 200
    cfg.eval_steps = 100
    cfg.save_image_steps = 0

    path = sys.argv[1] if len(sys.argv) > 1 else "media/images/anime-1_2k.png"
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = to_tensor(img_np, cfg.downsample, device)
    C, H, W = target.shape
    grid = get_grid(H, W, device=device)
    grad_prob = gradient_map(target)

    g = build_gaussians(target, grad_prob, grid, cfg, device)
    print(f"訓練：{H}x{W}, 起始 {g.num_gaussians} -> 目標 {cfg.num_gaussians} 高斯")
    t0 = time.time()
    g = train(g, target, grid, cfg)
    print(f"完成，耗時 {time.time()-t0:.1f}s，最終 {g.num_gaussians} 高斯")
    with torch.no_grad():
        pred = render.process(g, H, W, grid, cfg)
    print(f"最終 PSNR = {psnr(pred.clamp(0,1), target):.2f} dB")
    save_image(pred, f"{cfg.out_dir}/_check_train.png")
    save_image(target, f"{cfg.out_dir}/_check_train_target.png")
    print(f"已存 {cfg.out_dir}/_check_train.png 與 _check_train_target.png（比對重建 vs 原圖）")
