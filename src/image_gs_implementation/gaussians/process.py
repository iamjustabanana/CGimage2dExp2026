"""
Step 2: 高斯參數模型  ✅ 已完成
==============================
定義「一組 2D 高斯」這個可學習物件，並用梯度引導 + 取色做初始化。

每個高斯 4 組參數（全是 nn.Parameter，才會被 autograd / optimizer 接管）：
  xy    [N, 2]  位置(像素座標)
  scale [N, 2]  兩軸尺度(像素)
  rot   [N, 1]  旋轉角(弧度)
  feat  [N, C]  顏色

初始化（cfg.init_mode）：
  "gradient": 用 gradient_map 當機率取樣位置 -> 細節處高斯多；
              cfg.init_random_ratio 比例仍隨機散佈(避免平坦區沒高斯)。
  "random"  : 全部隨機(對照組)。
  顏色一律從原圖「該位置像素顏色」初始化，收斂更快。

對應官方：image-gs/model.py 的 _init_gaussians / _init_pos_scale_feat /
          _sample_pos / _get_target_features
"""

from __future__ import annotations

import torch
import torch.nn as nn


def process(target, grad_prob, grid, cfg, device) -> "Gaussians2D":
    """Step 2 主函式：建立並初始化一組高斯。

    target[C,H,W] / grad_prob[H*W] / grid[H*W,2] / cfg / device -> Gaussians2D
    """
    g = Gaussians2D(cfg.num_gaussians, feat_dim=target.shape[0], device=device)
    g.init_from_image(target, grad_prob, grid, cfg)
    return g


class Gaussians2D(nn.Module):
    def __init__(self, num_gaussians: int, feat_dim: int, device: str):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feat_dim = feat_dim
        # 先建立 4 個參數(暫時值)；真正初值在 init_from_image 填。
        # 用 nn.Parameter 包起來，optimizer 才會去更新它們、autograd 才會追蹤梯度。
        self.xy = nn.Parameter(torch.zeros(num_gaussians, 2, device=device))      # 位置(像素)
        self.scale = nn.Parameter(torch.ones(num_gaussians, 2, device=device))    # 兩軸尺度(inverse_scale 時存 1/s)
        self.rot = nn.Parameter(torch.zeros(num_gaussians, 1, device=device))     # 旋轉角(弧度)
        self.feat = nn.Parameter(torch.zeros(num_gaussians, feat_dim, device=device))  # 顏色

    @torch.no_grad()  # 初始化只是「填值」，不需要梯度
    def init_from_image(self, target, grad_prob, grid, cfg) -> None:
        """target[C,H,W] / grad_prob[H*W] / grid[H*W,2] / cfg。決定每個高斯初值。"""
        C, H, W = target.shape
        N = self.num_gaussians
        num_pixels = H * W
        device = self.xy.device

        # ---- 位置：選 N 個像素當高斯中心 ----
        if cfg.init_mode == "gradient":
            # 一部分隨機散佈(避免平坦區完全沒高斯)，其餘照梯度機率取樣(細節處密)
            num_random = round(cfg.init_random_ratio * N)
            idx_random = torch.randint(num_pixels, (num_random,), device=device)
            idx_grad = torch.multinomial(grad_prob, N - num_random, replacement=False)
            selected = torch.cat([idx_random, idx_grad])
        else:  # "random"：全部均勻隨機(對照組)
            selected = torch.randint(num_pixels, (N,), device=device)

        # grid 與 target 都是 row-major 展平，用同一組 selected 索引 -> 位置與顏色自動對齊
        self.xy.copy_(grid[selected])                       # [N,2] 取對應像素中心座標

        # ---- 尺度 / 旋轉 ----
        # inverse_scale: 參數存 1/s(論文預設)，所以填 1/init_scale；否則直接填 init_scale。
        self.scale.fill_(1.0 / cfg.init_scale if cfg.inverse_scale else cfg.init_scale)
        self.rot.zero_()                                    # 初始不旋轉

        # ---- 顏色：取原圖「該位置像素」的顏色 ----
        target_flat = target.reshape(C, num_pixels).permute(1, 0)  # [H*W, C]
        self.feat.copy_(target_flat[selected])              # [N,C]

    def forward(self):
        """回傳目前的 (xy, scale, rot, feat) 給渲染器。"""
        return self.xy, self.scale, self.rot, self.feat


@torch.no_grad()
def visualize_positions(gaussians: "Gaussians2D", target: torch.Tensor,
                        dim: float = 0.3, radius: int | None = None) -> torch.Tensor:
    """把高斯中心畫成紅點疊在(壓暗的)原圖上，回傳 [C,H,W]，用來肉眼檢查分佈。

    radius: 每個點畫成 (2r+1)² 的小方塊；不給就依解析度自動縮放(高解析才看得到點)。
    """
    C, H, W = target.shape
    if radius is None:
        radius = max(0, round(min(H, W) / 512) - 1)        # 512->0, 1024->1, 2048->3
    viz = target.clone() * dim                              # 原圖壓暗當底
    xc = gaussians.xy[:, 0].round().long()                 # 位置 -> 像素索引
    yc = gaussians.xy[:, 1].round().long()
    for dx in range(-radius, radius + 1):                  # 把點加粗成小方塊
        for dy in range(-radius, radius + 1):
            xs = (xc + dx).clamp(0, W - 1)
            ys = (yc + dy).clamp(0, H - 1)
            viz[0, ys, xs] = 1.0                           # 紅
            if C >= 3:
                viz[1, ys, xs] = 0.0
                viz[2, ys, xs] = 0.0
    return viz


# 自我驗證：uv run python -m src.image_gs_implementation.gaussians.process [圖片路徑]
if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image
    from ..config import load_config
    from ..input_image import to_tensor, get_grid, gradient_map, save_image

    cfg = load_config()
    path = sys.argv[1] if len(sys.argv) > 1 else "media/images/anime-1_2k.png"
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = to_tensor(img_np, cfg.downsample)
    C, H, W = target.shape
    grid = get_grid(H, W)
    grad_prob = gradient_map(target)

    g = process(target, grad_prob, grid, cfg, device="cpu")
    print(f"建立 {g.num_gaussians} 個高斯 (init_mode={cfg.init_mode})")
    print(f"  xy    {tuple(g.xy.shape)}  範圍 x[{g.xy[:,0].min():.0f},{g.xy[:,0].max():.0f}] y[{g.xy[:,1].min():.0f},{g.xy[:,1].max():.0f}]")
    print(f"  scale {tuple(g.scale.shape)}  值={g.scale[0].tolist()}")
    print(f"  rot   {tuple(g.rot.shape)}  feat {tuple(g.feat.shape)}")

    out = cfg.out_dir
    save_image(visualize_positions(g, target), f"{out}/_check_gaussians_{cfg.init_mode}.png")
    print(f"  已存高斯位置圖到 {out}/_check_gaussians_{cfg.init_mode}.png（紅點應集中在邊緣/細節）")
