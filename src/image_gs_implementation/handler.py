from __future__ import annotations

import numpy as np

from .input_image.process import process_batch as _process_batch


def process_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    # Future: add logging, validation, preprocessing, post-processing, etc.
    return _process_batch(images)
