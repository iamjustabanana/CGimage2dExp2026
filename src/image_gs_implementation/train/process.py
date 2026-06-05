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
      # (Step 5) 每隔 add_steps、且還沒加滿 -> progressive.process 補一批高斯，並重建 optimizer
      # (lr schedule) 沒進步就把 lr 除以 decay_ratio；衰減超過上限 -> 早停
  每 eval_steps 印 PSNR；每 save_image_steps 存圖。

要點：
  - 不同參數不同 lr(Adam param_groups)。
  - SSIM 先可省略(ratio=0 只用 L1)，或用純 PyTorch 簡版。
  - progressive 開啟時：起始只放 initial_ratio 的高斯，訓練中由 Step 5 漸進補滿；
    每次補完高斯數量會變，要用新的 Gaussians2D 重建 optimizer。

對應官方：image-gs/model.py optimize / _get_total_loss / _init_optimization / _lr_schedule
"""

from __future__ import annotations

from .. import render
from .. import progressive


def process(gaussians, target, grid, cfg):
    """Step 4 主函式：訓練並回傳訓練好的 gaussians（progressive 開啟時內部會呼叫 Step 5）。"""
    _, h, w = target.shape
    optimizer = make_optimizer(gaussians, cfg)
    # TODO(Step 4): 迴圈 render.process -> loss -> backward -> optimizer.step；定期評估/存圖。
    # TODO(Step 4): lr_schedule(沒進步衰減/早停)。
    # TODO(Step 5): progressive 開啟時，依 add_steps 呼叫 progressive.process 補高斯並重建 optimizer。
    raise NotImplementedError("Step 4 — 待實作：train process")


def make_optimizer(gaussians, cfg):
    """建 Adam，對 xy/scale/rot/feat 各設一組 lr。"""
    # TODO(Step 4): return torch.optim.Adam([{params, lr}, ...])
    raise NotImplementedError("Step 4 — 待實作：make_optimizer")
