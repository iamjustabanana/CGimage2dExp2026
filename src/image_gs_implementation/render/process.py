"""
Step 3: 可微分渲染器  ✅ 已完成（最核心，依論文 Eq.1/2/5）
=========================================================
把一組高斯「畫」成一張圖。整篇論文的心臟。

論文公式：
  Eq.1  G(x) = exp( -½ (x-μ)ᵀ Σ⁻¹ (x-μ) )
  Eq.2  Σ = R S Sᵀ Rᵀ        (因式分解保證半正定；S=diag(s₁,s₂)，R 為 θ 的旋轉矩陣)
  Eq.5  cr(x) = (1 / Σ_{i∈Sₖ} Gᵢ(x)) · Σ_{i∈Sₖ} Gᵢ(x)·cᵢ   (每像素取 top-K 後正規化加權平均)

關於 conic = Σ⁻¹（不用數值求逆）：
  因 R 正交(R⁻¹=Rᵀ)，Σ⁻¹ = R (S Sᵀ)⁻¹ Rᵀ = R · diag(1/s₁², 1/s₂²) · Rᵀ，直接解析得出。
  inverse_scale(論文預設)：參數存的是 1/s，故 1/s² = 參數²，conic = R·diag(參數²)·Rᵀ。

實作：
  3.1 全量：每像素對所有高斯算 G，正規化加權平均(把像素分塊避免爆顯存)。
  3.2 top-K：每像素依 G 值取前 K 個高斯再正規化(cfg.topk>0 啟用)。

對應官方：image-gs/model.py forward + gsplat rasterize_gaussians_sum(含 top-K)
"""

from __future__ import annotations

import torch


def build_conic(scale: torch.Tensor, rot: torch.Tensor, inverse_scale: bool) -> torch.Tensor:
    """由 scale[N,2]、rot[N,1] 算每個高斯的 conic = Σ⁻¹ [N,2,2]（論文 Eq.2 的解析逆）。

    inverse_scale=True：scale 存 1/s，1/s² = scale²。
    inverse_scale=False：scale 存 s，1/s² = 1/scale²。
    """
    # ── 由旋轉角 θ 組旋轉矩陣 R(Eq.2 的 R)──
    cos = torch.cos(rot)                                   # cos θ      [N,1]
    sin = torch.sin(rot)                                   # sin θ      [N,1]
    # R = [[cosθ, -sinθ],
    #      [sinθ,  cosθ]]   逐列拼出來再 stack 成 [N,2,2](stack 比就地賦值對 autograd 友善)
    row0 = torch.cat([cos, -sin], dim=-1)                  # R 第一列 [cosθ,-sinθ]  [N,2]
    row1 = torch.cat([sin, cos], dim=-1)                   # R 第二列 [sinθ, cosθ]  [N,2]
    R = torch.stack([row0, row1], dim=1)                   # [N,2,2]

    # ── 對角元素 = 1/s²(即 (S Sᵀ)⁻¹ 的對角)──
    # Eq.2: Σ = R S Sᵀ Rᵀ，S=diag(s₁,s₂) -> S Sᵀ=diag(s₁²,s₂²)
    # 因 R 正交(R⁻¹=Rᵀ)：Σ⁻¹ = R (S Sᵀ)⁻¹ Rᵀ = R·diag(1/s₁²,1/s₂²)·Rᵀ
    # inverse_scale: 參數本身就是 1/s，故 1/s² = 參數²；否則 1/s² = 1/參數²
    inv_var = scale**2 if inverse_scale else 1.0 / (scale**2)  # [N,2] = (1/s₁², 1/s₂²)
    D = torch.diag_embed(inv_var)                          # diag(1/s₁²,1/s₂²)  [N,2,2]

    # ── 組回 conic = Σ⁻¹(這行就是 Eq.2 取逆後的解析式)──
    return R @ D @ R.transpose(-1, -2)                     # Σ⁻¹ = R·diag(1/s²)·Rᵀ


def process(gaussians, h: int, w: int, grid, cfg) -> torch.Tensor:
    """Step 3 主函式：把高斯渲染成圖 [C,H,W]。

    gaussians: Gaussians2D(呼叫 gaussians() 取 xy/scale/rot/feat)；grid[H*W,2]。
    cfg.topk>0 走 top-K；<=0 或 >=N 走全量。
    """
    xy, scale, rot, feat = gaussians()                     # xy[N,2] scale[N,2] rot[N,1] feat[N,C]
    N, C = feat.shape
    conic = build_conic(scale, rot, cfg.inverse_scale)     # [N,2,2]

    P = grid.shape[0]                                      # 像素總數 = H*W
    out = torch.empty(P, C, device=feat.device, dtype=feat.dtype)

    use_topk = 0 < cfg.topk < N                            # K>=N 視為全量
    k = cfg.topk if use_topk else N

    # 把像素分塊：控制 [chunk, N] 權重矩陣大小，避免顯存爆掉(訓練時 autograd 還會留中間值)
    chunk = max(256, int(4_000_000 / N))

    for start in range(0, P, chunk):
        pts = grid[start:start + chunk]                    # x：這批像素座標 [B,2]
        # ── 算 Gᵢ(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))：論文 Eq.1 ──
        d = pts[:, None, :] - xy[None, :, :]               # [B,N,2] 像素到高斯中心的位移 = (x-μ)
        # 二次型 (x-μ)ᵀ Σ⁻¹ (x-μ)：tmp = d·conic，再跟 d 逐元素相乘求和
        tmp = torch.einsum("bni,nij->bnj", d, conic)       # [B,N,2]
        quad = (tmp * d).sum(-1)                            # [B,N]
        G = torch.exp(-0.5 * quad)                         # [B,N] = Gᵢ(x)，論文 Eq.1

        # ── 聚合顏色 cr(x) = Σ_{i∈Sₖ} Gᵢ·cᵢ / Σ_{i∈Sₖ} Gᵢ：論文 Eq.5(分子/分母分開算)──
        if use_topk:                                       # 論文 Eq.5：取 top-K 再正規化
            vals, idx = G.topk(k, dim=1)                   # [B,K] 每像素最大的 K 個 G 值
            feat_k = feat[idx]                             # [B,K,C] 對應顏色
            numer = (vals.unsqueeze(-1) * feat_k).sum(1)   # [B,C] = Σ G·c
            denom = vals.sum(1, keepdim=True)              # [B,1] = Σ G
        else:                                              # 全量(Eq.4 + 正規化)
            numer = G @ feat                               # [B,C]
            denom = G.sum(1, keepdim=True)                 # [B,1]

        out[start:start + chunk] = numer / (denom + cfg.eps)  # 正規化加權平均

    # [P,C] -> [H,W,C] -> [C,H,W]
    return out.reshape(h, w, C).permute(2, 0, 1).contiguous()


# 自我驗證：uv run python -m src.image_gs_implementation.render.process
# 手動擺幾個高斯，渲染應看到幾個彩色橢圓斑點。
if __name__ == "__main__":
    import torch as _t
    from ..config import load_config
    from ..gaussians import Gaussians2D
    from ..input_image import get_grid, save_image

    cfg = load_config()
    H = W = 256
    g = Gaussians2D(num_gaussians=4, feat_dim=3, device="cpu")
    std = _t.tensor([[20., 20.],     # 圓
                     [40., 10.],     # 橫長橢圓
                     [10., 40.],     # 直長橢圓
                     [30., 15.]])    # 斜橢圓(配旋轉)
    with _t.no_grad():
        g.xy.copy_(_t.tensor([[64., 64.], [192., 64.], [64., 192.], [192., 192.]]))
        # 依 inverse_scale 決定參數要存 1/std 還是 std
        g.scale.copy_(1.0 / std if cfg.inverse_scale else std)
        g.rot.copy_(_t.tensor([[0.], [0.], [0.], [3.14159 / 4]]))
        g.feat.copy_(_t.tensor([[1., 0., 0.],    # 紅
                                [0., 1., 0.],    # 綠
                                [0., 0., 1.],    # 藍
                                [1., 1., 0.]]))  # 黃
    grid = get_grid(H, W)
    img = process(g, H, W, grid, cfg)
    print(f"渲染輸出 shape = {tuple(img.shape)}  值域 = [{img.min():.3f}, {img.max():.3f}]")
    save_image(img, f"{cfg.out_dir}/_check_render.png")
    print(f"  已存 {cfg.out_dir}/_check_render.png（應看到 紅圓/綠橫橢圓/藍直橢圓/黃斜橢圓）")
