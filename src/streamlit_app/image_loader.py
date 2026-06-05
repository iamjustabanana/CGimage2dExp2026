from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load(source: str | bytes) -> np.ndarray:
    """Load an image from a local path or raw bytes into a numpy array (RGB)."""
    if isinstance(source, str):
        pil_img = Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        pil_img = Image.open(BytesIO(source)).convert("RGB")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    return np.asarray(pil_img, dtype=np.uint8)
