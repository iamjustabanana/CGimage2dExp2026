"""
Step 5: 誤差引導漸進最佳化 (progressive)  ⬜ 待實作
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
  - 把新舊高斯參數接起來(注意：要回傳新的 Gaussians2D，train 端要重建 optimizer)

對應官方：image-gs/model.py 的 _add_gaussians（與 optimize 內的呼叫時機）
"""

from __future__ import annotations

import torch


def num_to_add(current: int, cfg) -> int:
    """算這次該加幾個高斯(平均分配，且不超過 num_gaussians)。"""
    # TODO(Step 5): 依 add_times 把 (num_gaussians - 起始量) 平均分配。
    raise NotImplementedError("Step 5 — 待實作：num_to_add")


def process(gaussians, target, grid, cfg, device, add_num: int):
    """加一批 add_num 個高斯到高誤差區，回傳成長後的 Gaussians2D。"""
    # TODO(Step 5): render 目前圖 -> 誤差圖 -> 取樣新位置/顏色 -> 接上新參數。
    raise NotImplementedError("Step 5 — 待實作：progressive process")
