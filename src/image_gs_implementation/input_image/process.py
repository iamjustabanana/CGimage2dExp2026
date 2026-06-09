"""
Step 1: 輸入影像處理  ✅ 已完成
==============================
本子套件 = pipeline 的第一站，負責把「外面進來的 numpy 圖」轉成核心要用的東西。

  to_tensor    : np.uint8 RGB [H,W,C] -> torch tensor [C,H,W] (0~1)，可縮圖
  get_grid     : 每個像素中心的 (x,y) 座標 [H*W, 2]，單位 pixel
  gradient_map : Sobel 邊緣強度 -> 機率分布 [H*W]（Step 2 取樣初始位置用）
  psnr         : 重建品質(dB)

座標慣例（整個專案統一）：像素座標，x=欄(寬)、y=列(高)，像素(i,j)中心=(x=j+0.5, y=i+0.5)。
對應官方：image-gs/utils/image_utils.py 的 get_grid / compute_image_gradients / get_psnr

to_numpy / save_image 屬於通用 I/O，放在 utils/。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def process(image: np.ndarray, cfg, device: str = "cpu"):
    """Step 1 主函式：把外界進來的 numpy 圖轉成核心要用的三樣東西。

    回傳 (target, grid, grad_prob)：
      target    [C,H,W] torch 影像(0~1)
      grid      [H*W,2] 每像素座標
      grad_prob [H*W]   梯度機率圖(Step 2 取樣初始位置用)
    """
    target = to_tensor(image, cfg.downsample, device)   # numpy -> tensor
    _, h, w = target.shape
    grid = get_grid(h, w, device=device)
    grad_prob = gradient_map(target)
    return target, grid, grad_prob


def to_tensor(image: np.ndarray, downsample: int = 1, device: str = "cpu") -> torch.Tensor:
    """np.uint8 RGB [H,W,C] -> torch tensor [C,H,W]，值域 [0,1]。

    downsample>1 時用 area 模式縮小(縮圖最不失真)，例如 4 -> 1/4 邊長。
    """
    arr = image.astype(np.float32) / 255.0                      # 0~255 -> 0~1
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()    # [H,W,C] -> [C,H,W]
    if downsample > 1:
        t = F.interpolate(t.unsqueeze(0), scale_factor=1.0 / downsample,
                          mode="area").squeeze(0)
    return t.to(device)


def get_grid(h: int, w: int, device: str = "cpu") -> torch.Tensor:
    """每個像素中心的 (x,y) 座標 [H*W, 2]，row-major(與影像 reshape 對齊)。"""
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")     # 都是 [H,W]
    grid = torch.stack([grid_x, grid_y], dim=-1)               # [H,W,2]，最後一維 (x,y)
    return grid.reshape(-1, 2)


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    """Sobel 邊緣強度 -> 機率分布 [H*W]（已正規化，總和=1）。image: [C,H,W]。

    直覺：邊緣/細節大 -> 機率高 -> Step 2 在那裡多放高斯。
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)        # [1,1,H,W]
    # 用反射邊界補一圈再卷積(對齊scipy.ndimage.sobel 預設 mode='reflect')。
    # 若用 F.conv2d(padding=1) 會補 0，邊框會跟外面的 0 產生假跳變 -> 整圈假高梯度。
    gray = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(gray, kx)                                    # padding=0(已手動 pad)
    gy = F.conv2d(gray, ky)
    grad = torch.sqrt(gx**2 + gy**2).reshape(-1)              # ‖∇I(x)‖₂  Sobel 梯度大小 [H*W]
    grad = grad**2
    return grad / (grad.sum() + 1e-12)                        # ‖∇I(x)‖₂ / Σ_x ‖∇I(x)‖₂


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """重建品質(dB，越高越好)。pred/target: [C,H,W]，值域 0~1。"""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return (-10.0 * torch.log10(mse + 1e-12)).item()


# 自我驗證：uv run python -m src.image_gs_implementation.input_image.process [圖片路徑]
if __name__ == "__main__":
    import sys
    from PIL import Image
    from ..config import load_config
    from ..utils import save_image

    cfg = load_config()
    path = sys.argv[1] if len(sys.argv) > 1 else "media/images/anime-1_2k.png"
    img_np = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    target = to_tensor(img_np, cfg.downsample)
    C, H, W = target.shape
    print(f"載入 {path}")
    print(f"  影像 shape = {tuple(target.shape)}  值域 = [{target.min():.3f}, {target.max():.3f}]")

    grid = get_grid(H, W)
    print(f"  像素網格 shape = {tuple(grid.shape)}  第一點={grid[0].tolist()}  最後點={grid[-1].tolist()}")

    prob = gradient_map(target)
    print(f"  梯度機率圖 shape = {tuple(prob.shape)}  總和={prob.sum():.4f}（應≈1）")

    out = cfg.out_dir
    save_image(target, f"{out}/_check_input.png")
    save_image((prob / prob.max()).reshape(H, W).unsqueeze(0), f"{out}/_check_gradient.png")
    print(f"  已存圖到 {out}/_check_input.png 與 _check_gradient.png")
