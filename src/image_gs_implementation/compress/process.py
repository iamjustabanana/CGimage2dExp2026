"""
Step 6: 壓縮率計算 + 量化  ✅ 已完成
===================================
量化 = 把每個高斯參數從 float32 砍到較少 bit(例 8/12)，Image-GS「壓縮」的最後一塊。

主函式 process：回報壓縮率；cfg.quantize=True 時同時把高斯參數量化(就地)。

helper：
  compression_stats : 未壓縮 vs 壓縮的 bytes / bpp / 壓縮倍率。
      每個高斯 bit 數 = 2*pos_bits + 2*scale_bits + rot_bits + C*feat_bits
      (對照 image-gs/model.py:184)
  ste_quantize      : 帶直通估計(STE)的量化。
      前向：把 x 線性量化到 [0, 2^bits-1] 再還原為浮點；
      反向：梯度直接穿透(否則梯度=0、沒法訓練)。
      (對照 image-gs/utils/quantization_utils.py)

對應官方：image-gs/model.py _log_compression_rate / _quantize；
          image-gs/utils/quantization_utils.py ste_quantize
"""

from __future__ import annotations

import torch


# ── STE 量化 ──────────────────────────────────────────────────────────────────

def ste_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """帶 STE(直通估計)的量化。x 任意 shape，回傳同 shape。

    前向：把 x 線性映射到 [0, 2^bits-1] 的整數格，再還原回浮點值域。
    反向：梯度直接穿透(STE trick: x + (q - x).detach())，讓量化不斷掉梯度。

    參考論文：Straight-Through Estimator (arXiv:1308.3432)
    """
    qmin, qmax = 0, 2 ** bits - 1
    x_min = x.min().item()
    x_max = x.max().item()
    # scale：每個量化格對應多少浮點範圍
    scale = max((x_max - x_min) / (qmax - qmin), 1e-8)

    # ── 前向：量化(四捨五入到整數格) + 反量化(還原到浮點)──
    q = torch.round((x - x_min) / scale).clamp(qmin, qmax)  # 離散整數
    dq = q * scale + x_min                                   # 還原浮點

    # ── 反向：STE — 前向用量化值，反向梯度直通當作 identity ──
    # dq - x 在反向時梯度=0(detach)，所以整體梯度 = d(x)/dx = 1
    return x + (dq - x).detach()


# ── 壓縮率計算 ─────────────────────────────────────────────────────────────────

def compression_stats(num_gaussians: int, feat_dim: int, num_pixels: int, cfg) -> dict:
    """計算並印出未壓縮 vs 壓縮的大小，回傳資訊字典。

    未壓縮：假設原圖以 uint8 儲存 -> feat_dim × 8 bpp。
    壓縮：每個高斯 = (2*pos_bits + 2*scale_bits + rot_bits + C*feat_bits) bits。
    (公式對照 image-gs/model.py:184)
    """
    # ── 未壓縮(原始影像) ──
    bpp_uncompressed = float(feat_dim) * 8.0          # 每通道 8 bit
    bytes_uncompressed = num_pixels * feat_dim * 1.0  # uint8 = 1 byte per channel per pixel

    # ── 壓縮(高斯表示) ──
    # 每個高斯的 bit 數：位置(x,y) + 尺度(s1,s2) + 旋轉(θ) + 顏色(C 維)
    bits_per_gaussian = (
        2 * cfg.pos_bits
        + 2 * cfg.scale_bits
        + cfg.rot_bits
        + feat_dim * cfg.feat_bits
    )
    bits_compressed = bits_per_gaussian * num_gaussians
    bytes_compressed = bits_compressed / 8.0
    bpp_compressed = float(bits_compressed) / num_pixels

    compression_ratio = bpp_uncompressed / bpp_compressed if bpp_compressed > 0 else float("inf")

    print(f"  未壓縮 : {bytes_uncompressed/1e3:.2f} KB | {bpp_uncompressed:.1f} bpp")
    print(f"  壓縮   : {bytes_compressed/1e3:.2f} KB | {bpp_compressed:.3f} bpp  ({bits_per_gaussian} bits/高斯 × {num_gaussians} 高斯)")
    print(f"  壓縮率 : {compression_ratio:.2f}x  ({100.0 * bpp_compressed / bpp_uncompressed:.2f}% of original)")

    return {
        "num_gaussians": num_gaussians,
        "bpp_uncompressed": bpp_uncompressed,
        "bytes_uncompressed": bytes_uncompressed,
        "bpp_compressed": bpp_compressed,
        "bytes_compressed": bytes_compressed,
        "compression_ratio": compression_ratio,
    }


# ── 主函式 ────────────────────────────────────────────────────────────────────

def process(gaussians, num_pixels: int, cfg) -> dict:
    """Step 6 主函式：回報壓縮率；cfg.quantize=True 時同時量化高斯參數(就地)。

    注意：量化這裡是訓練後的「一次性量化」，把浮點參數截到設定 bit 數後存回。
    若要「訓練中量化感知」(QAT)，需在 render 的 forward 裡呼叫 ste_quantize —— 屬選配。
    """
    feat_dim = gaussians.feat.shape[1]
    num_g = gaussians.num_gaussians

    print(f"\n[Step 6] 壓縮率  ({num_g} 高斯 / {num_pixels} 像素):")
    stats = compression_stats(num_g, feat_dim, num_pixels, cfg)

    if cfg.quantize:
        print(f"  量化中 (pos={cfg.pos_bits}bit / scale={cfg.scale_bits}bit / rot={cfg.rot_bits}bit / feat={cfg.feat_bits}bit)...")
        # 訓練後就地量化：把 float32 參數截到設定 bit 數
        # _quantize 在 no_grad 下直接改值(對齊官方 _quantize)
        with torch.no_grad():
            gaussians.xy.copy_(ste_quantize(gaussians.xy, cfg.pos_bits))
            gaussians.scale.copy_(ste_quantize(gaussians.scale, cfg.scale_bits))
            gaussians.rot.copy_(ste_quantize(gaussians.rot, cfg.rot_bits))
            gaussians.feat.copy_(ste_quantize(gaussians.feat, cfg.feat_bits))
        print("  量化完成。")

    return stats


# ── 自我驗證 ───────────────────────────────────────────────────────────────────
# uv run python -m src.image_gs_implementation.compress.process
# 驗證：ste_quantize 精度損失合理；compression_stats 數字符合預期；量化前後 PSNR 變化小。
if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import sys
    from ..config import load_config, resolve_device
    from ..gaussians import process as build_gaussians
    from ..input_image import to_tensor, get_grid, gradient_map, psnr
    from ..render import process as do_render

    cfg = load_config()
    device = resolve_device(cfg.device)
    cfg.downsample = 8       # 快點跑：2k→256
    cfg.num_gaussians = 1000
    cfg.max_steps = 0        # 不訓練，只看量化對初始高斯的影響

    path = sys.argv[1] if len(sys.argv) > 1 else "media/images/anime-1_2k.png"
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = to_tensor(img_np, cfg.downsample, device)
    C, H, W = target.shape
    grid = get_grid(H, W, device=device)
    grad_prob = gradient_map(target)

    g = build_gaussians(target, grad_prob, grid, cfg, device)

    # ── 驗證一：ste_quantize 數值特性 ──
    t = g.feat.detach()
    for bits in (16, 12, 8, 4):
        qt = ste_quantize(t, bits)
        err = (qt - t).abs().mean().item()
        print(f"  feat ste_quantize({bits}bit): 平均誤差={err:.6f}")

    # ── 驗證二：compression_stats ──
    print()
    stats = compression_stats(g.num_gaussians, C, H * W, cfg)

    # ── 驗證三：量化前後 PSNR ──
    with torch.no_grad():
        pred_before = do_render(g, H, W, grid, cfg)
    psnr_before = psnr(pred_before.clamp(0, 1), target)

    cfg.quantize = True
    process(g, H * W, cfg)  # 就地量化

    with torch.no_grad():
        pred_after = do_render(g, H, W, grid, cfg)
    psnr_after = psnr(pred_after.clamp(0, 1), target)

    print(f"\n  PSNR 量化前={psnr_before:.2f} dB → 量化後={psnr_after:.2f} dB  (損失={psnr_before-psnr_after:.2f} dB)")
    print("  (只是初始化的高斯，數字低是正常的；量化損失才是重點)")
