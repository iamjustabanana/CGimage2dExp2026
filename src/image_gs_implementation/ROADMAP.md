# Image-GS 核心重實作 — 逐步路線圖

> 目標：用**純 PyTorch**自己重寫論文 *Image-GS: Content-Adaptive Image Representation via 2D Gaussians* (arXiv 2407.01866) 的**核心功能**：
> **把一張圖片轉成一組 2D 高斯，再用高斯壓縮圖片。**
>
> 刻意**不用**官方 CUDA `gsplat` / `fused-ssim` / `lpips`(那些是加速與額外指標，妨礙逐行理解)。慢沒關係，能 trace 最重要。
> 官方原始碼在 `image-gs/`(submodule)可隨時對照：`image-gs/model.py`、`image-gs/cfgs/default.yaml`。

---

## 專案結構（每個 Step = 一個子套件，對齊現有 `input_image/`）

```
src/image_gs_implementation/
├── __init__.py        # 對外暴露 process
├── config.py          # 【集中設定】所有超參數
├── handler.py         # pipeline 總指揮：process(np.uint8) -> list[np.uint8]
├── input_image/       # Step 1 ✅ 載圖/網格/梯度圖/PSNR/I-O
├── gaussians/         # Step 2 ⬜ 高斯參數模型 + 初始化
├── render/            # Step 3 ⬜ 可微分渲染器
├── train/             # Step 4 ⬜ 訓練迴圈
├── compress/          # Step 5 ⬜ 壓縮率 + 量化
└── outputs/           # 輸出圖(不進 git)
```

**資料介面慣例**：邊界用 **numpy(np.uint8 RGB)**(給 Streamlit)，內部用 **torch tensor**(需要 autograd + GPU)。

---

## 核心數學（整篇論文的心臟）

每個 2D 高斯有 4 組參數：位置 μ=(x,y)[2]、尺度 s=(s₁,s₂)[2]、旋轉 θ[1]、顏色 c[C]。

```
協方差   Σ   = R(θ) · diag(s₁², s₂²) · R(θ)ᵀ        R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]]
conic    Σ⁻¹
權重     w_k(p) = exp( -½ (p−μ_k)ᵀ Σ_k⁻¹ (p−μ_k) )

像素色（正規化加權平均，非 alpha 疊加）：
        Σ_k w_k(p) · c_k
 C(p) = ──────────────────
         Σ_k w_k(p) + ε
```
最簡版對「所有高斯」求和；論文用 **top-K**(每像素只取最近 K 個)加速，屬進階。

**為什麼能壓縮**：原圖存 `H×W×C` 個值；Image-GS 只存 `N` 個高斯 ×（2+2+1+C）。
`N` 遠小於像素數就壓縮了，再加**量化**(float32 → 8/12 bit)壓更多。公式見 `image-gs/model.py:184`。

座標慣例：像素座標，x=欄(寬)、y=列(高)，像素(i,j)中心=(x=j+0.5, y=i+0.5)。

---

## 步驟清單與進度

- [x] **Step 1 — `input_image/`**：`to_tensor`/`to_numpy`/`get_grid`/`gradient_map`/`psnr`/`save_image`
  - 驗證：`uv run python -m src.image_gs_implementation.input_image.process`
  - 看 `outputs/_check_gradient.png`，邊緣處應較亮。

- [ ] **Step 2 — `gaussians/`**：`Gaussians2D`(nn.Module，4 個 Parameter) + 梯度引導/取色初始化
  - 官方對照：`_init_gaussians` / `_init_pos_scale_feat` / `_sample_pos` / `_get_target_features`
  - 驗證：印參數 shape；把初始位置畫在圖上看是否集中在邊緣。

- [ ] **Step 3 — `render/`**：`build_conic` + `render`(全量加權平均、像素分塊)
  - 官方對照：`forward` / gsplat `rasterize_*`
  - 驗證：固定幾個高斯，渲染出彩色橢圓斑點。

- [ ] **Step 4 — `train/`**：`make_optimizer` + `train`(L1[+SSIM]、Adam 分組 lr)
  - 官方對照：`optimize` / `_get_total_loss` / `_init_optimization`
  - 驗證：loss 降、PSNR 升、輸出越來越像原圖。

- [ ] **Step 5 — `compress/`**：`compression_stats` + `ste_quantize`(STE 量化)
  - 官方對照：`_log_compression_rate` / `_quantize` / `utils/quantization_utils.py`
  - 驗證：報「壓縮 N 倍」；量化後 PSNR 掉一點但更小。

每完成一步：填滿該子套件 → 打開 `handler.py` 對應段落 → 來這份打勾。

### 進階（核心完成後再做，壓縮本身不需要）
top-K 正規化、progressive 漸進加高斯(`_add_gaussians`)、tile 加速、saliency 初始化。

---

## 常用指令
```bash
# Step 1 自我驗證
uv run python -m src.image_gs_implementation.input_image.process [圖片路徑]

# 跑整條 pipeline / 開 Streamlit 前端
uv run python -m src                      # 無參數 -> 開 Streamlit
uv run python -m src media/images/anime-1_2k.png   # 有圖 -> 跑 process

# 資料集
media/images/    media/textures/
```
