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


def _aggregate(pts: torch.Tensor, xy: torch.Tensor, conic: torch.Tensor,
               feat: torch.Tensor, topk: int) -> torch.Tensor:
    """核心：對一批像素 pts[B,2]，用給定的一組高斯算出顏色 [B,C]。

    這就是論文 Eq.1(權重) + Eq.5(top-K 正規化加權平均)，含數值穩定。
    全量版與 tile 版都呼叫它，差別只在「傳進來的高斯是全部、還是只有附近的」。
    """
    n = xy.shape[0]
    # ── 算二次型 q = (x-μ)ᵀ Σ⁻¹ (x-μ)，Eq.1 裡 exp 內的部分 ──
    d = pts[:, None, :] - xy[None, :, :]                   # [B,n,2] = (x-μ)
    tmp = torch.einsum("bni,nij->bnj", d, conic)           # (x-μ)·Σ⁻¹  [B,n,2]
    quad = (tmp * d).sum(-1)                                # [B,n]  q=(x-μ)ᵀΣ⁻¹(x-μ)

    # ── top-K：權重最大 = 二次型 q 最小的 K 個(論文 Eq.5)──
    use_topk = 0 < topk < n
    if use_topk:
        quad, idx = quad.topk(topk, dim=1, largest=False)  # [B,K] 最小的 K 個 q / 索引
        feat_sel = feat[idx]                               # [B,K,C] 對應顏色
    else:
        feat_sel = None

    # ── 數值穩定：每像素減最小 q(=最大權重)再 exp，最大權重=1、分母≥1(不黑、不用 eps)──
    q_min = quad.min(dim=1, keepdim=True).values           # [B,1]
    weight = torch.exp(-0.5 * (quad - q_min))              # [B,K或n]

    # ── 聚合 cr(x) = Σ w·c / Σ w(常數因子已約掉)──
    if use_topk:
        numer = (weight.unsqueeze(-1) * feat_sel).sum(1)   # [B,C]
    else:
        numer = weight @ feat                              # [B,C]
    denom = weight.sum(1, keepdim=True)                    # [B,1]
    return numer / denom


def _render_full(xy, conic, feat, h, w, grid, cfg) -> torch.Tensor:
    """全量 all-pairs：每像素對「所有」高斯算。把像素分塊避免峰值記憶體爆掉。"""
    N, C = feat.shape
    P = grid.shape[0]
    out = torch.empty(P, C, device=feat.device, dtype=feat.dtype)
    chunk = max(256, int(4_000_000 / N))                   # 控制 [chunk,N] 大小
    for start in range(0, P, chunk):
        out[start:start + chunk] = _aggregate(grid[start:start + chunk], xy, conic, feat, cfg.topk)
    return out.reshape(h, w, C).permute(2, 0, 1).contiguous()


def _render_tiled(xy, scale, conic, feat, h, w, cfg) -> torch.Tensor:
    """tile 渲染：把圖切成 T×T 方塊，每塊只用「footprint 有碰到它」的高斯。

    剔除遠處高斯後，每像素只跟少數高斯算 -> 計算量與記憶體都大降(論文 tile 做法)。
    沒有任何高斯碰到的 tile 會留黑(罕見；訓練時靠 progressive 補)。
    """
    C = feat.shape[1]
    device = feat.device
    T = cfg.tile_size
    # 每個高斯的 3σ 包圍半徑(取較長軸)：決定它的 footprint 會碰到哪些 tile
    std = (1.0 / scale) if cfg.inverse_scale else scale    # [N,2] 實際 std
    radius = 3.0 * std.max(dim=1).values                   # [N]
    cx, cy = xy[:, 0], xy[:, 1]

    out = torch.zeros(h, w, C, device=device, dtype=feat.dtype)
    nty, ntx = (h + T - 1) // T, (w + T - 1) // T
    for ty in range(nty):
        y0, y1 = ty * T, min((ty + 1) * T, h)
        for tx in range(ntx):
            x0, x1 = tx * T, min((tx + 1) * T, w)
            # 剔除：留下 footprint 方框 [cx±r,cy±r] 與此 tile [x0,x1)×[y0,y1) 有交集的高斯
            m = (cx + radius >= x0) & (cx - radius < x1) & (cy + radius >= y0) & (cy - radius < y1)
            idx = m.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue                                   # 無高斯 -> 留黑
            # 此 tile 的像素座標(中心)
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
            xs = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([gx, gy], dim=-1).reshape(-1, 2)   # [th*tw,2]
            color = _aggregate(pts, xy[idx], conic[idx], feat[idx], cfg.topk)
            out[y0:y1, x0:x1] = color.reshape(y1 - y0, x1 - x0, C)
    return out.permute(2, 0, 1).contiguous()


def process(gaussians, h: int, w: int, grid, cfg) -> torch.Tensor:
    """Step 3 主函式：把高斯渲染成圖 [C,H,W]。

    cfg.tile_size>0 走 tile(快)，否則走全量 all-pairs。cfg.topk>0 取 top-K。
    """
    xy, scale, rot, feat = gaussians()                     # xy[N,2] scale[N,2] rot[N,1] feat[N,C]
    conic = build_conic(scale, rot, cfg.inverse_scale)     # [N,2,2]
    if cfg.tile_size and cfg.tile_size > 0:
        return _render_tiled(xy, scale, conic, feat, h, w, cfg)
    return _render_full(xy, conic, feat, h, w, grid, cfg)


# 自我驗證：uv run python -m src.image_gs_implementation.render.process
# 手動擺幾個高斯，渲染應看到幾個彩色橢圓斑點。
if __name__ == "__main__":
    import torch as _t
    from ..config import load_config
    from ..gaussians import Gaussians2D
    from ..input_image import get_grid
    from ..utils import save_image

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
