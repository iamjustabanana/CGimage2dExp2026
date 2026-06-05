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
├── gaussians/         # Step 2 ✅ 高斯參數模型 + 初始化
├── render/            # Step 3 ✅ 可微分渲染器(全量 + top-K)
├── train/             # Step 4 ✅ 訓練迴圈(+ lr 衰減/早停)
├── progressive/       # Step 5 ✅ 誤差引導漸進加高斯
├── compress/          # Step 6 ⬜ 壓縮率 + 量化
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
先做全量(對所有高斯求和)，再做論文的 **top-K**(每像素只取權重最大的 K 個)：除了加速，也讓遠處高斯不互相污染、邊界更銳利。**top-K 與 progressive 都是論文核心**，本計畫完整實作。

**為什麼能壓縮**：原圖存 `H×W×C` 個值；Image-GS 只存 `N` 個高斯 ×（2+2+1+C）。
`N` 遠小於像素數就壓縮了，再加**量化**(float32 → 8/12 bit)壓更多。公式見 `image-gs/model.py:184`。

座標慣例：像素座標，x=欄(寬)、y=列(高)，像素(i,j)中心=(x=j+0.5, y=i+0.5)。

---

## 步驟清單與進度

- [x] **Step 1 — `input_image/`**：`to_tensor`/`to_numpy`/`get_grid`/`gradient_map`/`psnr`/`save_image`
  - 驗證：`uv run python -m src.image_gs_implementation.input_image.process`
  - 看 `outputs/_check_gradient.png`，邊緣處應較亮。

- [x] **Step 2 — `gaussians/`**：`Gaussians2D`(nn.Module，4 個 Parameter) + 梯度引導/取色初始化
  - 官方對照：`_init_gaussians` / `_init_pos_scale_feat` / `_sample_pos` / `_get_target_features`
  - 驗證：`uv run python -m src.image_gs_implementation.gaussians.process`
    看 `outputs/_check_gaussians_gradient.png`，紅點(高斯中心)應集中在邊緣。

- [x] **Step 3 — `render/`**：`build_conic` + `render`(依論文 Eq.1/2/5)
  - 3.1 全量加權平均(像素分塊避免爆顯存)
  - 3.2 **top-K 正規化**(論文核心)：每像素依 G 值取最大的 K 個高斯
  - conic 用因式分解解析式 `R·diag(1/s²)·Rᵀ`(不用數值求逆);參數存 inverse scale 1/s
  - 官方對照：`forward` / gsplat `rasterize_gaussians_sum`(含 top-K)
  - 驗證：`uv run python -m src.image_gs_implementation.render.process`(看 `_check_render.png`
    紅圓/綠橫/藍直/黃斜橢圓);handler 多輸出 `render_init.png`(未訓練初始高斯的重建)。
  - 數值穩定：聚合前每像素「減去最小二次型(=最大權重)」再 exp(類 softmax)，最大權重恆=1、
    分母≥1。**不可用 `+eps` 當分母保護**：權重會 underflow，eps 反而蓋過真實權重 -> 整片變黑洞。
  - 注意：純 PyTorch 全量 all-pairs + autograd 記憶體吃重，訓練(Step 4)時要靠
    downsample / 控制高斯數;預覽渲染記得包 `torch.no_grad()`。

- [x] **Step 4 — `train/`**：`make_optimizer` + `train` + 純 PyTorch `ssim`
  - loss = L1 + ratio·(1-SSIM)；Adam 分組 lr；clip θ∈[0,π]、scale>0；lr 衰減/早停。
  - 內含 Step 5 整合：每 add_steps 步呼叫 `progressive.process` 補高斯並重建 optimizer。
  - 官方對照：`optimize` / `_get_total_loss` / `_init_optimization` / `_lr_schedule`
  - 驗證：`uv run python -m src.image_gs_implementation.train.process`
    256px/2000 高斯/800 步 → PSNR 23→28.6 dB，重建幾乎無法分辨；每次 progressive 補點 PSNR 跳升。
  - ⚠ 純 PyTorch 全量很慢(256px 800 步 ≈ 9 min)+ 吃記憶體；高解析需調大 `downsample`。
    要加速/上高解析 → 需 tile-based 或 gradient checkpointing(見下方選配)。

- [x] **Step 5 — `progressive/`**：`num_to_add` + `process`(誤差引導漸進加高斯)
  - 渲染目前圖 → 誤差圖(機率) → multinomial 取樣新位置 → 新高斯顏色=殘差 → 接上舊的回傳。
  - 起始只放 `initial_ratio`；訓練中分 `add_times` 次在高誤差區補高斯；由 train 迴圈呼叫並重建 optimizer。
  - 官方對照：`_add_gaussians`(及 `optimize` 內呼叫時機)
  - 驗證：`uv run python -m src.image_gs_implementation.progressive.process`
    新高斯處平均誤差 ≈ 全圖 2.3 倍；`_check_progressive.png` 紅點落在高誤差(亮)區。

- [ ] **Step 6 — `compress/`**：`compression_stats` + `ste_quantize`(STE 量化)
  - 官方對照：`_log_compression_rate` / `_quantize` / `utils/quantization_utils.py`
  - 驗證：報「壓縮 N 倍」；量化後 PSNR 掉一點但更小。

每完成一步：填滿該子套件 → 打開 `handler.py` 對應段落 → 來這份打勾。

### 已加的優化
- [x] **tile 渲染**(`render/_render_tiled`)：切 T×T 方塊，每塊只用 footprint 有交集的高斯。
  純 PyTorch 最佳 `tile_size≈64`；512px 渲染快 ~20x、256px 訓練快 ~8x，PSNR 不變。
  `cfg.tile_size=0` 可切回全量 all-pairs 對照。

### 仍屬選配（暫不做）
- gradient checkpointing(進一步省訓練記憶體，讓高解析訓練不 OOM)
- saliency 初始化(需 EML-Net 預訓練模型，額外下載)

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
