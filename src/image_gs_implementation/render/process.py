"""
Step 3: 可微分渲染器  ⬜ 待實作（最核心）
========================================
把一組高斯「畫」成一張圖。整篇論文的心臟。

數學：
  1. 協方差   Σ   = R(θ) · diag(s₁², s₂²) · R(θ)ᵀ
  2. conic    Σ⁻¹                                    (2x2 反矩陣)
  3. 權重     w_k(p) = exp(-½ (p-μ_k)ᵀ Σ_k⁻¹ (p-μ_k))
  4. 像素色（正規化加權平均，非 alpha 疊加）：
       C(p) = Σ_k w_k c_k / (Σ_k w_k + eps)

實作要點（分兩階段做）：
  3.1 全量版：每像素對所有高斯，用 broadcasting；像素分塊(chunk)避免爆顯存。
  3.2 top-K 正規化(論文核心)：每像素只取「權重最大的 K 個」高斯做加權平均。
      cfg.topk>0 啟用；除了加速，也讓遠處高斯不互相污染、邊界更銳利。
      純 PyTorch 做法：算完每像素對所有高斯的權重後取 topk(沿高斯維度)，
      只對那 K 個做正規化加權平均。

對應官方：image-gs/model.py forward + gsplat rasterize_gaussians_sum(內含 top-K)
"""

from __future__ import annotations

import torch


def process(gaussians, h: int, w: int, grid, cfg) -> torch.Tensor:
    """Step 3 主函式：把高斯渲染成圖 [C,H,W]。

    gaussians: Gaussians2D(呼叫 gaussians() 取 xy/scale/rot/feat)；grid[H*W,2]。
    cfg.topk>0 走 top-K 正規化；=0 走全量加權平均。
    """
    xy, scale, rot, feat = gaussians()
    # TODO(Step 3.1): conic=build_conic(scale,rot)；分塊算 w 與加權平均；reshape 回 [C,H,W]。
    # TODO(Step 3.2): cfg.topk>0 時，每像素沿高斯維取 topk 權重再正規化加權平均。
    raise NotImplementedError("Step 3 — 待實作：render process")


def build_conic(scale: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """scale[N,2]、rot[N,1] -> 每個高斯的 Σ⁻¹(conic) [N,2,2]。"""
    # TODO(Step 3): 組 R、Σ = R diag(s²) Rᵀ，再求逆。
    raise NotImplementedError("Step 3 — 待實作：build_conic")
