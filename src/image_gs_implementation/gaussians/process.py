"""
Step 2: 高斯參數模型  ⬜ 待實作
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
        # TODO(Step 2): 建立 4 個 nn.Parameter(xy/scale/rot/feat)，先給暫時值，
        #               真正初值在 init_from_image 填。
        raise NotImplementedError("Step 2 — 待實作：建立高斯參數")

    @torch.no_grad()
    def init_from_image(self, target, grad_prob, grid, cfg) -> None:
        """target[C,H,W] / grad_prob[H*W] / grid[H*W,2] / cfg。決定每個高斯初值。"""
        # TODO(Step 2): 依 init_mode 取樣位置；scale=init_scale；rot=0；feat=該位置顏色。
        raise NotImplementedError("Step 2 — 待實作：高斯初始化")

    def forward(self):
        """回傳目前的 (xy, scale, rot, feat) 給渲染器。"""
        return self.xy, self.scale, self.rot, self.feat
