"""
Step 4: 訓練迴圈  ⬜ 待實作
==========================
用梯度下降把高斯逼近目標圖（這就是「把圖片轉成高斯」的過程）。

流程：
  for step in range(max_steps):
      pred = render.process(gaussians, ...)       # 用目前高斯畫一張
      loss = L1(pred, target) + ratio*(1-SSIM)    # 差多少
      loss.backward()                             # autograd 算每個高斯梯度
      optimizer.step()                            # 微調高斯
  每 eval_steps 印 PSNR；每 save_image_steps 存圖。

要點：不同參數不同 lr(Adam param_groups)；SSIM 先可省略(ratio=0 只用 L1)。
對應官方：image-gs/model.py optimize / _get_total_loss / _init_optimization
"""

from __future__ import annotations

from .. import render


def process(gaussians, target, grid, cfg):
    """Step 4 主函式：訓練並回傳訓練好的 gaussians。"""
    _, h, w = target.shape
    optimizer = make_optimizer(gaussians, cfg)
    # TODO(Step 4): 迴圈 render.process -> loss -> backward -> optimizer.step；定期評估/存圖。
    raise NotImplementedError("Step 4 — 待實作：train process")


def make_optimizer(gaussians, cfg):
    """建 Adam，對 xy/scale/rot/feat 各設一組 lr。"""
    # TODO(Step 4): return torch.optim.Adam([{params, lr}, ...])
    raise NotImplementedError("Step 4 — 待實作：make_optimizer")
