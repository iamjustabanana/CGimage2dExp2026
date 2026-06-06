"""
誤差引導漸進最佳化 (progressive)  ✅ 已完成
=================================================
論文核心貢獻之一。訓練時不要一次放滿所有高斯，而是：
  1. 一開始只放 initial_ratio 比例的高斯。
  2. 每隔 add_steps 步，看目前重建圖「哪裡誤差大」，就在那裡補新的高斯。
  3. 分 add_times 次補到 num_gaussians 為止，之後再訓練 post_min_steps。

好處：高斯會自動往「難重建的細節區」集中 -> 同樣數量品質更好，
      而且自然形成 level-of-detail 階層(先粗後細)。

主函式 process：執行「加一批高斯」這個動作，回傳成長後的 Gaussians2D。
  - 用 render 算目前重建 -> 跟 target 比 -> 誤差圖
  - 以誤差為機率取樣新位置(對齊 Step 2 的取樣)
  - 新高斯顏色 = 該處殘差；尺度=init_scale；旋轉=0
  - 把新舊高斯參數接起來(回傳新的 Gaussians2D，train 端要重建 optimizer)

對應官方：image-gs/model.py 的 _add_gaussians（與 optimize 內的呼叫時機）
"""

from __future__ import annotations

import math

import torch

from ... import render
from ...gaussians import Gaussians2D


def num_to_add(current: int, cfg) -> int:
    """算這次該加幾個高斯(平均分配，且不超過 num_gaussians)。

    起始量 = ceil(initial_ratio × 總數)；要補的總量 = 總數 − 起始量，分 add_times 次。
    回傳每次的量，但夾到「還剩多少沒加」以免超過。
    """
    total = cfg.num_gaussians
    initial = math.ceil(cfg.initial_ratio * total)
    per_add = math.ceil((total - initial) / cfg.add_times)   # 每次平均補這麼多
    remaining = total - current                              # 還沒補的量
    return max(0, min(per_add, remaining))


@torch.no_grad()  # 加高斯是「決定初值」，不需要梯度
def process(gaussians, target, grid, cfg, device, add_num: int):
    """加一批 add_num 個高斯到高誤差區，回傳成長後的 Gaussians2D。"""
    if add_num <= 0:
        return gaussians
    C, H, W = target.shape
    num_pixels = H * W

    # 1. 目前重建(no_grad 由 decorator 提供) -> 殘差 -> 誤差圖
    pred = render.process(gaussians, H, W, grid, cfg).clamp(0, 1)   # [C,H,W]
    diff = target - pred                                           # [C,H,W] 殘差(含正負號)
    error = (diff.abs().mean(dim=0).reshape(-1)) ** 2              # [H*W] 誤差大小(平方放大)
    prob = error / (error.sum() + 1e-12)                          # 正規化成機率分布

    # 2. 在高誤差處取樣 add_num 個新位置(對齊 Step 2 用機率取樣的做法)
    selected = torch.multinomial(prob, add_num, replacement=False)  # [add_num] 像素索引

    # 3. 新高斯的參數
    new_xy = grid[selected]                                       # 位置 = 那些像素中心 [add,2]
    fill = 1.0 / cfg.init_scale if cfg.inverse_scale else cfg.init_scale
    new_scale = torch.full((add_num, 2), fill, device=device)    # 尺度 = init_scale
    new_rot = torch.zeros(add_num, 1, device=device)             # 旋轉 = 0
    diff_flat = diff.permute(1, 2, 0).reshape(num_pixels, C)     # [H*W,C]
    new_feat = diff_flat[selected]                               # 顏色 = 該處殘差 [add,C]

    # 4. 接上舊高斯，建一個成長後的新 Gaussians2D(train 端會用它重建 optimizer)
    grown = Gaussians2D(gaussians.num_gaussians + add_num, C, device)
    grown.xy.copy_(torch.cat([gaussians.xy, new_xy], dim=0))
    grown.scale.copy_(torch.cat([gaussians.scale, new_scale], dim=0))
    grown.rot.copy_(torch.cat([gaussians.rot, new_rot], dim=0))
    grown.feat.copy_(torch.cat([gaussians.feat, new_feat], dim=0))
    return grown


# 自我驗證：uv run python -m src.image_gs_implementation.train.progressive.process
# 先用少量高斯(重建較差)，加一批，檢查新點是否落在高誤差區。
if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image
    from ...config import load_config, resolve_device
    from ...gaussians import Gaussians2D as _G
    from ...input_image import to_tensor, get_grid, gradient_map
    from ...utils import save_image

    cfg = load_config()
    device = resolve_device(cfg.device)
    path = sys.argv[1] if len(sys.argv) > 1 else "media/images/anime-1_2k.png"
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = to_tensor(img_np, cfg.downsample, device)
    C, H, W = target.shape
    grid = get_grid(H, W, device=device)
    grad_prob = gradient_map(target)

    # 故意只用少量高斯 -> 重建差 -> 誤差明顯
    g = _G(num_gaussians=600, feat_dim=C, device=device)
    g.init_from_image(target, grad_prob, grid, cfg)
    add = 600
    g2 = process(g, target, grid, cfg, device, add)
    print(f"高斯數量：{g.num_gaussians} -> {g2.num_gaussians}（加了 {add}）")
    print(f"num_to_add(目前={g.num_gaussians}) = {num_to_add(g.num_gaussians, cfg)}")

    # 視覺化：誤差圖(灰) + 新高斯位置(紅)
    with torch.no_grad():
        pred = render.process(g, H, W, grid, cfg).clamp(0, 1)
    err = (target - pred).abs().mean(dim=0)                       # [H,W] 誤差大小
    err_norm = (err / (err.max() + 1e-12)).unsqueeze(0).expand(3, H, W)
    save_image(pred, f"{cfg.out_dir}/_check_progressive_render.png")
    save_image(err_norm, f"{cfg.out_dir}/_check_progressive_error.png")
    vis = err_norm.clone()
    new_xy = g2.xy[g.num_gaussians:]                             # 只取新加的
    xs = new_xy[:, 0].round().long().clamp(0, W - 1)
    ys = new_xy[:, 1].round().long().clamp(0, H - 1)

    # 量化檢查：新高斯所在像素的誤差 vs 全圖平均誤差
    err_flat = err.reshape(-1)
    new_err = err_flat[ys * W + xs].mean().item()
    print(f"  全圖平均誤差={err_flat.mean():.5f}，新高斯處平均誤差={new_err:.5f}（後者應明顯較大）")

    r = max(2, round(min(H, W) / 256))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            vis[0, (ys + dy).clamp(0, H - 1), (xs + dx).clamp(0, W - 1)] = 1.0
            vis[1, (ys + dy).clamp(0, H - 1), (xs + dx).clamp(0, W - 1)] = 0.0
            vis[2, (ys + dy).clamp(0, H - 1), (xs + dx).clamp(0, W - 1)] = 0.0
    save_image(vis, f"{cfg.out_dir}/_check_progressive.png")
    print(f"  已存 {cfg.out_dir}/_check_progressive.png（灰=誤差，紅點=新高斯，應落在亮處）")
