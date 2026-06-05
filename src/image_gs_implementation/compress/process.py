"""
Step 6: 壓縮率計算 + 量化  ⬜ 待實作
===================================
量化 = 把每個高斯參數從 float32 砍到較少 bit(例 8/12)，Image-GS「壓縮」的最後一塊。

主函式 process：(可選)量化高斯 + 回報壓縮率。
helper：
  compression_stats : 未壓縮 vs 壓縮 bytes / bpp / 壓縮倍率。
      每張高斯 bit 數 = 2*pos_bits + 2*scale_bits + rot_bits + C*feat_bits
      (對照 image-gs/model.py:184)
  ste_quantize      : 帶 STE(直通估計)的量化。
      前向：量化到 2^bits 個離散階；反向：梯度直接穿過(否則梯度=0、沒法訓練)。

對應官方：image-gs/model.py _log_compression_rate / _quantize；
          image-gs/utils/quantization_utils.py ste_quantize
"""

from __future__ import annotations


def process(gaussians, num_pixels: int, cfg) -> dict:
    """Step 5 主函式：回報壓縮率(若 cfg.quantize 則同時量化高斯參數)。"""
    feat_dim = gaussians.feat.shape[1]
    stats = compression_stats(cfg.num_gaussians, feat_dim, num_pixels, cfg)
    # TODO(Step 5): if cfg.quantize: 用 ste_quantize 量化 gaussians 的各參數。
    return stats


def compression_stats(num_gaussians: int, feat_dim: int, num_pixels: int, cfg) -> dict:
    """回傳壓縮資訊(bytes、bpp、壓縮倍率)並印出。"""
    # TODO(Step 5): 依公式算未壓縮/壓縮大小。
    raise NotImplementedError("Step 5 — 待實作：compression_stats")


def ste_quantize(x, bits: int):
    """帶 straight-through estimator 的量化。x 任意 shape，回傳同 shape。"""
    # TODO(Step 5): q=round(x 正規化到[0,2^bits-1])再還原；x+(q-x).detach() 直通梯度。
    raise NotImplementedError("Step 5 — 待實作：ste_quantize")
