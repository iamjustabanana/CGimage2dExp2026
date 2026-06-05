from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
    __package__ = "src.streamlit_app"
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import streamlit as st

from .. import image_gs_implementation
from .image_loader import load, ALLOWED_EXTENSIONS

st.set_page_config(page_title="Image Processor", layout="wide")
st.title("Image Processor")

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

if st.button("Upload & Process", disabled=not uploaded_files):
    all_results = []
    for f in uploaded_files:
        arr = load(f.read())
        all_results.extend(image_gs_implementation.process(arr))
    st.session_state.results = all_results
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
