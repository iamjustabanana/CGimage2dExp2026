from __future__ import annotations

import glob
import json
import os
import sys
import threading
from collections import defaultdict

if __name__ == "__main__" and __package__ is None:
    __package__ = "src.streamlit_app"
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import streamlit as st
from PIL import Image

from .. import image_gs_implementation
from ..image_gs_implementation.config import load_config
from .image_loader import load, ALLOWED_EXTENSIONS

# ── Module-level training state（執行緒安全的共享狀態）─────────────────────────
_training_state: dict = {"done": False, "results": None, "error": None}


def _run_training(arr):
    _training_state.update({"done": False, "results": None, "error": None})
    try:
        _training_state["results"] = image_gs_implementation.process(arr)
    except Exception as e:
        _training_state["error"] = str(e)
    finally:
        _training_state["done"] = True


# ── 頁面設定 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Image Processor", layout="wide")
st.title("Image Processor")

# ── Session state 初始化 ───────────────────────────────────────────────────────
for key, default in [("results", []), ("slide_index", 0), ("_step_idx", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 上傳 + 觸發訓練 ────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Choose images",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    accept_multiple_files=True,
    key="file_uploader",
)

is_training = (
    st.session_state.get("_thread") is not None
    and st.session_state["_thread"].is_alive()
)

if st.button("Upload & Process", disabled=not uploaded_files or is_training):
    arr = load(uploaded_files[0].read())
    t = threading.Thread(target=_run_training, args=(arr,), daemon=True)
    t.start()
    st.session_state.update({"_thread": t, "results": [], "slide_index": 0, "_step_idx": 0})
    st.rerun()

# 訓練完成 -> 把結果搬進 session_state（在主執行緒執行，安全）
if not is_training and _training_state["done"] and _training_state["results"] is not None:
    st.session_state.results = _training_state["results"]
    _training_state.update({"done": False, "results": None})
    st.rerun()

if _training_state.get("error"):
    st.error(f"訓練錯誤：{_training_state['error']}")

# ── Results Slideshow ──────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.subheader("Results Slideshow")

    total = len(st.session_state.results)
    idx = st.session_state.slide_index

    col_prev, col_img, col_next = st.columns([1, 6, 1])

    with col_prev:
        st.button("◀ Prev", on_click=lambda: st.session_state.update(
            slide_index=(st.session_state.slide_index - 1) % total
        ), width='stretch')

    with col_img:
        st.image(
            st.session_state.results[idx],
            caption=f"Output {idx + 1} / {total}",
            width='stretch',
        )

    with col_next:
        st.button("Next ▶", on_click=lambda: st.session_state.update(
            slide_index=(st.session_state.slide_index + 1) % total
        ), width='stretch')

    if st.button("Clear"):
        st.session_state.results = []
        st.session_state.slide_index = 0
        st.rerun()

# ── 即時指標（progress bar + PSNR/Loss/N）────────────────────────────────────
@st.fragment(run_every=2 if is_training else None)
def _show_live_metrics():
    _cfg = load_config()
    progress_path = os.path.join(_cfg.out_dir, "steps", "progress.json")
    if not os.path.exists(progress_path):
        if is_training:
            st.info("訓練啟動中，等待第一次 eval...")
        return
    with open(progress_path) as f:
        p = json.load(f)
    pct = p["step"] / p["total_steps"] if p["total_steps"] else 0
    label = "完成" if p.get("status") == "done" else f"Step {p['step']} / {p['total_steps']}"
    st.progress(min(pct, 1.0), text=label)
    c1, c2, c3 = st.columns(3)
    c1.metric("PSNR", f"{p['psnr']} dB")
    c2.metric("Loss", f"{p['loss']:.4f}")
    c3.metric("Gaussians", p["num_gaussians"])


# ── 步驟瀏覽器（左右切換 + 多視角並排）────────────────────────────────────────
@st.fragment(run_every=2 if is_training else None)
def _show_progress():
    _cfg = load_config()
    steps_dir = os.path.join(_cfg.out_dir, "steps")
    json_files = sorted(glob.glob(os.path.join(steps_dir, "step*.json")))
    if not json_files:
        return

    # 讀所有 JSON，依 step 分組
    groups: dict[int, list[dict]] = defaultdict(list)
    for jf in json_files:
        png_path = jf.replace(".json", ".png")
        if not os.path.exists(png_path):
            continue  # PNG 尚未寫完，跳過
        with open(jf) as f:
            meta = json.load(f)
        meta["img_path"] = png_path
        groups[meta["step"]].append(meta)

    step_keys = sorted(groups.keys())
    total_steps = len(step_keys)
    if total_steps == 0:
        return

    # 只有當用戶「停在最後一步」時，新步驟出現才自動跟過去；否則保留用戶的導航位置
    last_total = st.session_state.get("_last_total_steps", 0)
    if total_steps > last_total and st.session_state["_step_idx"] >= last_total - 1:
        st.session_state["_step_idx"] = total_steps - 1
    st.session_state["_last_total_steps"] = total_steps

    idx = max(0, min(st.session_state["_step_idx"], total_steps - 1))

    st.divider()
    st.subheader("Training Progress")

    # 導航列
    col_prev, col_info, col_next = st.columns([1, 5, 1])
    with col_prev:
        if st.button("◀ Prev", disabled=(idx == 0), key="_step_prev"):
            st.session_state["_step_idx"] = idx - 1
    with col_next:
        if st.button("Next ▶", disabled=(idx == total_steps - 1), key="_step_next"):
            st.session_state["_step_idx"] = idx + 1

    step_num = step_keys[idx]
    items = sorted(groups[step_num], key=lambda x: x["view"])
    first = items[0]
    event_label = {"init": "初始化", "add": "＋補高斯", "periodic": "定期"}.get(first["event"], first["event"])

    with col_info:
        st.caption(
            f"**Step {step_num}** ({idx + 1}/{total_steps})  |  {event_label}"
            f"  |  PSNR {first['psnr']} dB  |  Loss {first['loss']}"
            f"  |  N={first['num_gaussians']}"
        )

    # 多視角並排
    cols = st.columns(max(len(items), 1))
    for col, item in zip(cols, items):
        with col:
            st.image(Image.open(item["img_path"]),
                     caption=item["view"],
                     use_container_width=True)


_show_live_metrics()
_show_progress()
