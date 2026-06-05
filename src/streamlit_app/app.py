from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
    __package__ = "src.streamlit_app"
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import streamlit as st

from ..input_image.process import process_batch
from .image_loader import load, ALLOWED_EXTENSIONS

st.set_page_config(page_title="Image Batch Processor", layout="wide")
st.title("Image Batch Processor")

if "batch" not in st.session_state:
    st.session_state.batch = []
if "results" not in st.session_state:
    st.session_state.results = []
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

uploaded_files = st.file_uploader(
    "Choose images",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    accept_multiple_files=True,
    key="file_uploader",
)

col_up, _ = st.columns([1, 5])
with col_up:
    if st.button("Upload", disabled=not uploaded_files, use_container_width=True):
        for f in uploaded_files:
            st.session_state.batch.append(load(f.read()))
        st.success(f"Loaded {len(uploaded_files)} image(s)")
        st.rerun()

st.divider()
st.write(f"**Batch size:** {len(st.session_state.batch)}")

for i, arr in enumerate(st.session_state.batch):
    st.image(arr, caption=f"Image {i + 1}  ({arr.shape[1]}×{arr.shape[0]})", width=200)

if st.button("Process Batch", disabled=len(st.session_state.batch) == 0):
    with st.spinner("Processing..."):
        st.session_state.results = process_batch(st.session_state.batch)
    st.session_state.slide_index = 0
    st.rerun()

if st.session_state.results:
    st.divider()
    st.subheader("Results Slideshow")

    total = len(st.session_state.results)
    idx = st.session_state.slide_index

    col_prev, col_img, col_next = st.columns([1, 6, 1])

    with col_prev:
        st.button("◀ Prev", on_click=lambda: st.session_state.update(
            slide_index=(st.session_state.slide_index - 1) % total
        ), use_container_width=True)

    with col_img:
        st.image(
            st.session_state.results[idx],
            caption=f"Output {idx + 1} / {total}",
            use_container_width=True,
        )

    with col_next:
        st.button("Next ▶", on_click=lambda: st.session_state.update(
            slide_index=(st.session_state.slide_index + 1) % total
        ), use_container_width=True)

    if st.button("Clear & Start Over"):
        st.session_state.batch = []
        st.session_state.results = []
        st.session_state.slide_index = 0
        st.rerun()
